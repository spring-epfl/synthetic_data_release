"""
Command-line interface for running privacy evaluation with respect to the risk of linkability
"""

import json

from os import mkdir, path
from numpy.random import choice, seed
from argparse import ArgumentParser
from pandas import DataFrame

from utils.datagen import load_s3_data_as_df, load_local_data_as_df
from utils.utils import json_numpy_serialzer
from utils.logging import LOGGER
from utils.constants import *

from feature_sets.independent_histograms import HistogramFeatureSet
from feature_sets.model_agnostic import NaiveFeatureSet, EnsembleFeatureSet
from feature_sets.bayes import CorrelationsFeatureSet

from sanitisation_techniques.sanitiser import SanitiserNHS

from generative_models.ctgan import CTGAN
from generative_models.pate_gan import PATEGAN
from generative_models.data_synthesiser import (IndependentHistogram,
                                                BayesianNet,
                                                PrivBayes)

from attack_models.mia_classifier import (MIAttackClassifierRandomForest,
                                          generate_mia_shadow_data,
                                          generate_mia_anon_data)

from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)
simplefilter('ignore', category=DeprecationWarning)

cwd = path.dirname(__file__)


SEED = 42


def main():
    argparser = ArgumentParser()
    datasource = argparser.add_mutually_exclusive_group()
    datasource.add_argument('--s3name', '-S3', type=str, choices=['adult', 'census', 'credit', 'alarm', 'insurance'], help='Name of the dataset to run on')
    datasource.add_argument('--datapath', '-D', type=str, help='Relative path to cwd of a local data file')
    argparser.add_argument('--runconfig', '-RC', default='runconfig_mia.json', type=str, help='Path relative to cwd of runconfig file')
    argparser.add_argument('--outdir', '-O', default='tests', type=str, help='Path relative to cwd for storing output files')
    args = argparser.parse_args()

    # Load runconfig
    with open(path.join(cwd, args.runconfig)) as f:
        runconfig = json.load(f)
    print('Runconfig:')
    print(runconfig)

    # Load data
    if args.s3name is not None:
        rawPop, metadata = load_s3_data_as_df(args.s3name)
        dname = args.s3name
    else:
        rawPop, metadata = load_local_data_as_df(path.join(cwd, args.datapath))
        dname = args.datapath.split('/')[-1]

    print(f'Loaded data {dname}:')
    print(rawPop.info())

    # Make sure outdir exists
    if not path.isdir(args.outdir):
        mkdir(args.outdir)

    seed(SEED)

    ########################
    #### GAME INPUTS #######
    ########################
    # Pick targets
    targetIDs = choice(list(rawPop.index), size=runconfig['nTargets'], replace=False).tolist()

    # If specified: Add specific target records
    if runconfig['Targets'] is not None:
        targetIDs.extend(runconfig['Targets'])

    targets = rawPop.loc[targetIDs, :]

    # Drop targets from population
    rawPopDropTargets = rawPop.drop(targetIDs)

    # Init adversary's prior knowledge
    rawAidx = choice(list(rawPopDropTargets.index), size=runconfig['sizeRawA'], replace=False).tolist()
    rawA = rawPop.loc[rawAidx, :]

    # List of candidate generative models to evaluate
    gmList = []
    if 'generativeModels' in runconfig.keys():
        for gm, paramsList in runconfig['generativeModels'].items():
            if gm == 'IndependentHistogram':
                for params in paramsList:
                    gmList.append(IndependentHistogram(metadata, *params))
            elif gm == 'BayesianNet':
                for params in paramsList:
                    gmList.append(BayesianNet(metadata, *params))
            elif gm == 'PrivBayes':
                for params in paramsList:
                    gmList.append(PrivBayes(metadata, *params))
            elif gm == 'CTGAN':
                for params in paramsList:
                    gmList.append(CTGAN(metadata, *params))
            elif gm == 'PATEGAN':
                for params in paramsList:
                    gmList.append(PATEGAN(metadata, *params))
            else:
                raise ValueError(f'Unknown GM {gm}')

    # List of candidate sanitisation techniques to evaluate
    sanList = []
    if 'sanitisationTechniques' in runconfig.keys():
        for name, paramsList in runconfig['sanitisationTechniques'].items():
            if name == 'SanitiserNHS':
                for params in paramsList:
                    sanList.append(SanitiserNHS(metadata, *params))
            else:
                raise ValueError(f'Unknown sanitisation technique {name}')

    ###################################
    #### ATTACK TRAINING #############
    ##################################
    print('\n---- Attack training ----')
    attacks = {}

    for tid in targetIDs:
        print(f'\n--- Adversary picks target {tid} ---')
        target = targets.loc[[tid]]
        attacks[tid] = {}

        for San in sanList:
            LOGGER.info(f'Start: Attack training for {San.__name__}...')

            attacks[tid][San.__name__] = {}

            # Generate example datasets for training attack classifier
            sanA, labelsA = generate_mia_anon_data(San, target, rawA, runconfig['sizeRawT'], runconfig['nShadows'] * runconfig['nSynA'])

            # Train attack on shadow data
            for Feature in [NaiveFeatureSet(DataFrame),
                            HistogramFeatureSet(DataFrame, metadata, nbins=San.histogram_size, quids=San.quids),
                            CorrelationsFeatureSet(DataFrame, metadata, quids=San.quids),
                            EnsembleFeatureSet(DataFrame, metadata, nbins=San.histogram_size, quasi_id_cols=San.quids)]:

                Attack = MIAttackClassifierRandomForest(metadata=metadata, FeatureSet=Feature, quids=San.quids)
                Attack.train(sanA, labelsA)
                attacks[tid][San.__name__][f'{Feature.__name__}'] = Attack

            # Clean up
            del sanA, labelsA

            LOGGER.info(f'Finished: Attack training.')

        for GenModel in gmList:
            LOGGER.info(f'Start: Attack training for {GenModel.__name__}...')

            attacks[tid][GenModel.__name__] = {}

            # Generate shadow model data for training attacks on this target
            if 'PrivBay' in GenModel.__name__:
                print('Use BayNet as shadow')
                ShadowModel = BayesianNet(metadata,
                                          histogram_bins=GenModel.histogram_bins,
                                          degree=GenModel.degree,
                                          infer_ranges=GenModel.infer_ranges,
                                          multiprocess=GenModel.multiprocess,
                                          seed=GenModel.seed)
                synA, labelsSA = generate_mia_shadow_data(ShadowModel, target, rawA, runconfig['sizeRawT'], runconfig['sizeSynT'], runconfig['nShadows'], runconfig['nSynA'])

            else:
                synA, labelsSA = generate_mia_shadow_data(GenModel, target, rawA, runconfig['sizeRawT'], runconfig['sizeSynT'], runconfig['nShadows'], runconfig['nSynA'])

            # Train attack on shadow data
            for Feature in [NaiveFeatureSet(GenModel.datatype), HistogramFeatureSet(GenModel.datatype, metadata),
                            CorrelationsFeatureSet(GenModel.datatype, metadata), EnsembleFeatureSet(GenModel.datatype, metadata)]:
                Attack  = MIAttackClassifierRandomForest(metadata, Feature)
                Attack.train(synA, labelsSA)
                attacks[tid][GenModel.__name__][f'{Feature.__name__}'] = Attack

            # Clean up
            del synA, labelsSA

            LOGGER.info(f'Finished: Attack training.')

    ##################################
    ######### EVALUATION #############
    ##################################
    resultsTargetPrivacy = {tid: {gm.__name__: {} for gm in gmList + sanList} for tid in targetIDs}

    print('\n---- Start the game ----')
    for nr in range(runconfig['nIter']):
        print(f'\n--- Game iteration {nr + 1} ---')
        # Draw a raw dataset
        rIdx = choice(list(rawPopDropTargets.index), size=runconfig['sizeRawT'], replace=False).tolist()
        rawTout = rawPopDropTargets.loc[rIdx]

        for GenModel in gmList:
            LOGGER.info(f'Start: Evaluation for model {GenModel.__name__}...')
            # Train a generative model
            GenModel.fit(rawTout)
            synTwithoutTarget = [GenModel.generate_samples(runconfig['sizeSynT']) for _ in range(runconfig['nSynT'])]
            synLabelsOut = [LABEL_OUT for _ in range(runconfig['nSynT'])]

            for tid in targetIDs:
                LOGGER.info(f'Target: {tid}')
                target = targets.loc[[tid]]
                resultsTargetPrivacy[tid][f'{GenModel.__name__}'][nr] = {}

                rawTin = rawTout.append(target)
                GenModel.fit(rawTin)
                synTwithTarget = [GenModel.generate_samples(runconfig['sizeSynT']) for _ in range(runconfig['nSynT'])]
                synLabelsIn = [LABEL_IN for _ in range(runconfig['nSynT'])]

                synT = synTwithoutTarget + synTwithTarget
                synTlabels = synLabelsOut + synLabelsIn

                # Run attacks
                for feature, Attack in attacks[tid][f'{GenModel.__name__}'].items():
                    # Produce a guess for each synthetic dataset
                    attackerGuesses = Attack.attack(synT)

                    resDict = {
                        'Secret': synTlabels,
                        'AttackerGuess': attackerGuesses
                    }
                    resultsTargetPrivacy[tid][f'{GenModel.__name__}'][nr][feature] = resDict

            del synT, synTwithoutTarget, synTwithTarget

            LOGGER.info(f'Finished: Evaluation for model {GenModel.__name__}.')

        for San in sanList:
            LOGGER.info(f'Start: Evaluation for sanitiser {San.__name__}...')
            sanOut = San.sanitise(rawTout)

            for tid in targetIDs:
                LOGGER.info(f'Target: {tid}')
                target = targets.loc[[tid]]
                resultsTargetPrivacy[tid][San.__name__][nr] = {}

                rawTin = rawTout.append(target)
                sanIn = San.sanitise(rawTin)

                sanT = [sanOut, sanIn]
                sanTLabels = [LABEL_OUT, LABEL_IN]

                # Run attacks
                for feature, Attack in attacks[tid][San.__name__].items():
                    # Produce a guess for each synthetic dataset
                    attackerGuesses = Attack.attack(sanT, attemptLinkage=True, target=target)

                    resDict = {
                        'Secret': sanTLabels,
                        'AttackerGuess': attackerGuesses
                    }
                    resultsTargetPrivacy[tid][San.__name__][nr][feature] = resDict

            del sanT, sanOut, sanIn

            LOGGER.info(f'Finished: Evaluation for model {San.__name__}.')

    outfile = f"ResultsMIA_{dname}"
    LOGGER.info(f"Write results to {path.join(f'{args.outdir}', f'{outfile}')}")

    with open(path.join(f'{args.outdir}', f'{outfile}.json'), 'w') as f:
        json.dump(resultsTargetPrivacy, f, indent=2, default=json_numpy_serialzer)


if __name__ == "__main__":
    main()