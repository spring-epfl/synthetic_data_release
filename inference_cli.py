"""
Command-line interface for running privacy evaluation under an attribute inference adversary
"""

import json

from os import mkdir, path
from numpy.random import choice, seed
from argparse import ArgumentParser

from utils.datagen import load_s3_data_as_df, load_local_data_as_df
from utils.utils import json_numpy_serialzer
from utils.logging import LOGGER
from utils.constants import *

from generative_models.ctgan import CTGAN
from generative_models.data_synthesiser import IndependentHistogram, BayesianNet, PrivBayes
from generative_models.pate_gan import PATEGAN
from sanitisation_techniques.sanitiser import SanitiserNHS
from attack_models.reconstruction import LinRegAttack, RandForestAttack

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

    ##################################
    ######### EVALUATION #############
    ##################################
    resultsTargetPrivacy = {tid: {sa: {gm.__name__: {} for gm in gmList + sanList} for sa in runconfig['sensitiveAttributes']} for tid in targetIDs}
    # Add entry for raw
    for tid in targetIDs:
        for sa in runconfig['sensitiveAttributes']:
            resultsTargetPrivacy[tid][sa]['Raw'] = {}

    print('\n---- Start the game ----')
    for nr in range(runconfig['nIter']):
        print(f'\n--- Game iteration {nr + 1} ---')
        # Draw a raw dataset
        rIdx = choice(list(rawPopDropTargets.index), size=runconfig['sizeRawT'], replace=False).tolist()
        rawTout = rawPopDropTargets.loc[rIdx]

        ###############
        ## ATTACKS ####
        ###############
        attacks = {}
        for sa, atype in runconfig['sensitiveAttributes'].items():
            if atype == 'LinReg':
                attacks[sa] = LinRegAttack(sensitiveAttribute=sa, metadata=metadata)
            elif atype == 'Classification':
                attacks[sa] = RandForestAttack(sensitiveAttribute=sa, metadata=metadata)

        #### Assess advantage raw
        for sa, Attack in attacks.items():
            Attack.train(rawTout)

            for tid in targetIDs:
                target = targets.loc[[tid]]
                targetAux = target.loc[[tid], Attack.knownAttributes]
                targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                guess = Attack.attack(targetAux, attemptLinkage=True, data=rawTout)
                pCorrect = Attack.get_likelihood(targetAux, targetSecret, attemptLinkage=True, data=rawTout)

                resultsTargetPrivacy[tid][sa]['Raw'][nr] = {
                    'AttackerGuess': [guess],
                    'ProbCorrect': [pCorrect],
                    'TargetPresence': [LABEL_OUT]
                }

        for tid in targetIDs:
            target = targets.loc[[tid]]
            rawTin = rawTout.append(target)

            for sa, Attack in attacks.items():
                targetAux = target.loc[[tid], Attack.knownAttributes]
                targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                guess = Attack.attack(targetAux, attemptLinkage=True, data=rawTin)
                pCorrect = Attack.get_likelihood(targetAux, targetSecret, attemptLinkage=True, data=rawTin)

                resultsTargetPrivacy[tid][sa]['Raw'][nr]['AttackerGuess'].append(guess)
                resultsTargetPrivacy[tid][sa]['Raw'][nr]['ProbCorrect'].append(pCorrect)
                resultsTargetPrivacy[tid][sa]['Raw'][nr]['TargetPresence'].append(LABEL_IN)

        ##### Assess advantage Syn
        for GenModel in gmList:
            LOGGER.info(f'Start: Evaluation for model {GenModel.__name__}...')
            GenModel.fit(rawTout)
            synTwithoutTarget = [GenModel.generate_samples(runconfig['sizeSynT']) for _ in range(runconfig['nSynT'])]

            for sa, Attack in attacks.items():
                for tid in targetIDs:
                    resultsTargetPrivacy[tid][sa][GenModel.__name__][nr] = {
                        'AttackerGuess': [],
                        'ProbCorrect': [],
                        'TargetPresence': [LABEL_OUT for _ in range(runconfig['nSynT'])]
                    }

                for syn in synTwithoutTarget:
                    Attack.train(syn)

                    for tid in targetIDs:
                        target = targets.loc[[tid]]
                        targetAux = target.loc[[tid], Attack.knownAttributes]
                        targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                        guess = Attack.attack(targetAux)
                        pCorrect = Attack.get_likelihood(targetAux, targetSecret)

                        resultsTargetPrivacy[tid][sa][GenModel.__name__][nr]['AttackerGuess'].append(guess)
                        resultsTargetPrivacy[tid][sa][GenModel.__name__][nr]['ProbCorrect'].append(pCorrect)

            del synTwithoutTarget

            for tid in targetIDs:
                LOGGER.info(f'Target: {tid}')
                target = targets.loc[[tid]]
                rawTin = rawTout.append(target)

                GenModel.fit(rawTin)
                synTwithTarget = [GenModel.generate_samples(runconfig['sizeSynT']) for _ in range(runconfig['nSynT'])]

                for sa, Attack in attacks.items():
                    targetAux = target.loc[[tid], Attack.knownAttributes]
                    targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                    for syn in synTwithTarget:
                        Attack.train(syn)

                        guess = Attack.attack(targetAux)
                        pCorrect = Attack.get_likelihood(targetAux, targetSecret)

                        resultsTargetPrivacy[tid][sa][GenModel.__name__][nr]['AttackerGuess'].append(guess)
                        resultsTargetPrivacy[tid][sa][GenModel.__name__][nr]['ProbCorrect'].append(pCorrect)
                        resultsTargetPrivacy[tid][sa][GenModel.__name__][nr]['TargetPresence'].append(LABEL_IN)
            del synTwithTarget

        for San in sanList:
            LOGGER.info(f'Start: Evaluation for sanitiser {San.__name__}...')
            attacks = {}
            for sa, atype in runconfig['sensitiveAttributes'].items():
                if atype == 'LinReg':
                    attacks[sa] = LinRegAttack(sensitiveAttribute=sa, metadata=metadata, quids=San.quids)
                elif atype == 'Classification':
                    attacks[sa] = RandForestAttack(sensitiveAttribute=sa, metadata=metadata, quids=San.quids)

            sanOut = San.sanitise(rawTout)

            for sa, Attack in attacks.items():
                Attack.train(sanOut)

                for tid in targetIDs:
                    target = targets.loc[[tid]]
                    targetAux = target.loc[[tid], Attack.knownAttributes]
                    targetSecret = target.loc[tid, Attack.sensitiveAttribute]

                    guess = Attack.attack(targetAux, attemptLinkage=True, data=sanOut)
                    pCorrect = Attack.get_likelihood(targetAux, targetSecret, attemptLinkage=True, data=sanOut)

                    resultsTargetPrivacy[tid][sa][San.__name__][nr] = {
                        'AttackerGuess': [guess],
                        'ProbCorrect': [pCorrect],
                        'TargetPresence': [LABEL_OUT]
                }

            for tid in targetIDs:
                LOGGER.info(f'Target: {tid}')
                target = targets.loc[[tid]]
                rawTin = rawTout.append(target)
                sanIn = San.sanitise(rawTin)

                for sa, Attack in attacks.items():
                    targetAux = target.loc[[tid], Attack.knownAttributes]
                    targetSecret = target.loc[tid, Attack.sensitiveAttribute]


                    Attack.train(sanIn)

                    guess = Attack.attack(targetAux, attemptLinkage=True, data=sanIn)
                    pCorrect = Attack.get_likelihood(targetAux, targetSecret, attemptLinkage=True, data=sanIn)

                    resultsTargetPrivacy[tid][sa][San.__name__][nr]['AttackerGuess'].append(guess)
                    resultsTargetPrivacy[tid][sa][San.__name__][nr]['ProbCorrect'].append(pCorrect)
                    resultsTargetPrivacy[tid][sa][San.__name__][nr]['TargetPresence'].append(LABEL_IN)

    outfile = f"ResultsMLEAI_{dname}"
    LOGGER.info(f"Write results to {path.join(f'{args.outdir}', f'{outfile}')}")

    with open(path.join(f'{args.outdir}', f'{outfile}.json'), 'w') as f:
        json.dump(resultsTargetPrivacy, f, indent=2, default=json_numpy_serialzer)

if __name__ == "__main__":
    main()