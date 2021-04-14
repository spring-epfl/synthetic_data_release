"""
Command-line interface for running privacy evaluation under a membership inference adversary
"""

import json
from argparse import ArgumentParser
from os import path

from numpy import arange
from numpy.random import choice

from synthetic_data.utils.datagen import load_local_data_as_df
from synthetic_data.utils.utils import json_numpy_serialzer, setup_logger
from synthetic_data.utils.evaluation_framework import (
    craft_outlier,
    evaluate_mia,
)

from synthetic_data.feature_sets.independent_histograms import HistogramFeatureSet
from synthetic_data.feature_sets.model_agnostic import NaiveFeatureSet, EnsembleFeatureSet
from synthetic_data.feature_sets.bayes import CorrelationsFeatureSet

from synthetic_data.generative_models.ctgan import CTGAN
from synthetic_data.generative_models.data_synthesiser import IndependentHistogram, BayesianNet, PrivBayes
from synthetic_data.generative_models.pate_gan import PateGan

from synthetic_data.privacy_attacks.membership_inference import (
    LABEL_IN,
    LABEL_OUT,
    MIAttackClassifierKNN,
    MIAttackClassifierLinearSVC,
    MIAttackClassifierLogReg,
    MIAttackClassifierRandomForest,
    MIAttackClassifierSVC,
)

from warnings import filterwarnings
filterwarnings('ignore')

cwd = path.dirname(__file__)

LOGGER = setup_logger()


def main():
    """Entrypoint of the program."""

    argparser = ArgumentParser()
    argparser.add_argument('--datapath', '-D', type=str, help='Relative path to cwd of a local data file')
    argparser.add_argument('--attack_model', '-AM', type=str, default='ANY', choices=['RandomForest', 'LogReg', 'LinearSVC', 'SVC', 'KNN', 'ANY'])
    argparser.add_argument('--runconfig', '-RC', default='runconfig_mia.json', type=str, help='Path relative to cwd of runconfig file')
    argparser.add_argument('--outdir', '-O', default='outputs/test', type=str, help='Path relative to cwd for storing output files')
    args = argparser.parse_args()

    # Load runconfig
    with open(path.join(cwd, args.runconfig)) as f:
        runconfig = json.load(f)
    print('Runconfig:')
    print(runconfig)

    # Load data
    RawDF, metadata = load_local_data_as_df(path.join(cwd, args.datapath))
    dname = args.datapath.split('/')[-1]
    RawDF['ID'] = [f'ID{i}' for i in arange(len(RawDF))]
    RawDF = RawDF.set_index('ID')

    LOGGER.info(f'Loaded data {dname}:')
    print(RawDF.info())

    # Randomly select nt target records T = (t_1, ..., t_(nt))
    targetIDs = choice(list(RawDF.index), size=runconfig['nTargets'], replace=False).tolist()
    Targets = RawDF.loc[targetIDs, :]

    # Drop targets from sample population
    RawDFdropT = RawDF.drop(targetIDs)

    # Add a crafted outlier target to the evaluation set
    targetCraft = craft_outlier(RawDF, runconfig['sizeTargetCraft'])
    targetIDs.extend(list(set(targetCraft.index)))
    Targets = Targets.append(targetCraft)

    # Sample adversary's background knowledge RawA
    rawAidx = choice(list(RawDFdropT.index), size=runconfig['sizeRawA'], replace=False).tolist()

    # Sample k independent target test sets
    rawTindices = [choice(list(RawDFdropT.index), size=runconfig['sizeRawT'], replace=False).tolist() for nr in range(runconfig['nIter'])]

    # List of candidate generative models to evaluate
    gmList = []
    for gm, paramsList in runconfig['generativeModels'].items():
        if gm == 'IndependentHistogram':
            for params in paramsList:
                gmList.append(IndependentHistogram(*params))
        elif gm == 'BayesianNet':
            for params in paramsList:
                gmList.append(BayesianNet(*params))
        elif gm == 'PrivBayes':
            for params in paramsList:
                gmList.append(PrivBayes(*params))
        elif gm == 'CTGAN':
            for params in paramsList:
                gmList.append(CTGAN(metadata, *params))
        elif gm == 'PateGan':
            for params in paramsList:
                gmList.append(PateGan(metadata, *params))
        else:
            raise ValueError(f'Unknown GM {gm}')

    LOGGER.info('Start: Privacy evaluation...')
    for GenModel in gmList:
        print(f'-- Target Model: {GenModel.__name__} --')

        FeatureList = [NaiveFeatureSet(GenModel.datatype), HistogramFeatureSet(GenModel.datatype, metadata), CorrelationsFeatureSet(GenModel.datatype, metadata), EnsembleFeatureSet(GenModel.datatype, metadata)]

        prior = {LABEL_IN: runconfig['prior']['IN'], LABEL_OUT: runconfig['prior']['OUT']}

        if args.attack_model == 'RandomForest':
            AttacksList = [MIAttackClassifierRandomForest(metadata, prior, F) for F in FeatureList]
        elif args.attack_model == 'LogReg':
            AttacksList = [MIAttackClassifierLogReg(metadata, prior, F) for F in FeatureList]
        elif args.attack_model == 'LinearSVC':
            AttacksList = [MIAttackClassifierLinearSVC(metadata, prior, F) for F in FeatureList]
        elif args.attack_model == 'SVC':
            AttacksList = [MIAttackClassifierSVC(metadata, prior, F) for F in FeatureList]
        elif args.attack_model == 'KNN':
            AttacksList = [MIAttackClassifierKNN(metadata, prior, F) for F in FeatureList]
        elif args.attack_model == 'ANY':
            AttacksList = []
            for F in FeatureList:
                AttacksList.extend([MIAttackClassifierRandomForest(metadata, prior, F),
                                    MIAttackClassifierLogReg(metadata, prior, F),
                                    MIAttackClassifierKNN(metadata, prior, F)])
        else:
            raise ValueError(f'Unknown AM {args.attack_model}')

        # Run privacy evaluation under MIA adversary
        results = evaluate_mia(GenModel, AttacksList, RawDFdropT, Targets, targetIDs, rawAidx, rawTindices,
                               runconfig['sizeRawT'], runconfig['sizeSynT'], runconfig['nSynT'],
                               runconfig['nSynA'], runconfig['nShadows'], metadata)

        outfile = f"{dname}{GenModel.__name__}MIA"

        LOGGER.info(f'Write results to {outfile}')

        with open(path.join(f'{args.outdir}', f'{outfile}.json'), 'w') as f:
            json.dump(results, f, indent=2, default=json_numpy_serialzer)

    LOGGER.info('Finished: Privacy evaluation')

if __name__ == "__main__":
    main()
