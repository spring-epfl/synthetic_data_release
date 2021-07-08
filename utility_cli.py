"""
Command-line interface for running utility evaluation
"""

import json

from os import mkdir, path
from numpy import mean
from numpy.random import choice, seed
from argparse import ArgumentParser

from utils.datagen import load_s3_data_as_df, load_local_data_as_df
from utils.utils import json_numpy_serialzer
from utils.logging import LOGGER
from utils.constants import *

from sanitisation_techniques.sanitiser import SanitiserNHS
from generative_models.data_synthesiser import BayesianNet, PrivBayes, IndependentHistogram
from generative_models.ctgan import CTGAN
from generative_models.pate_gan import PATEGAN
from predictive_models.predictive_model import RandForestClassTask, LogRegClassTask, LinRegTask

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
    argparser.add_argument('--outdir', '-O', default='outputs/test', type=str, help='Path relative to cwd for storing output files')
    args = argparser.parse_args()

    seed(SEED)
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

    ########################
    #### GAME INPUTS #######
    ########################
    # Train test split
    filterExpression = runconfig['dataFilter']
    rawTrain = rawPop[rawPop[filterExpression['train'][0]].str.contains(filterExpression['train'][1])]
    rawTest = rawPop[rawPop[filterExpression['test'][0]].str.contains(filterExpression['test'][1])]

    ## Game inputs
    # Pick targets
    targetIDs = choice(list(rawTrain.index), size=runconfig['nTargets'], replace=False).tolist()

    # If specified: Add specific target records
    if runconfig['Targets'] is not None:
        targetIDs.extend(runconfig['Targets'])

    targets = rawTrain.loc[targetIDs, :]

    # Drop targets from population
    rawTrainWoTargets = rawTrain.drop(targetIDs)

    # Get test target records
    testRecordIDs = choice(list(rawTest.index), size=runconfig['nTargets'], replace=False).tolist()

    # If specified: Add specific target records
    if runconfig['TestRecords'] is not None:
        testRecordIDs.extend(runconfig['TestRecords'])

    testRecords = rawTest.loc[testRecordIDs, :]

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

    utilityTasks = []
    for taskName, paramsList in runconfig['utilityTasks'].items():
        if taskName == 'RandForestClass':
            for params in paramsList:
                utilityTasks.append(RandForestClassTask(metadata, *params))
        elif taskName == 'LogRegClass':
            for params in paramsList:
                utilityTasks.append(LogRegClassTask(metadata, *params))
        elif taskName == 'LinReg':
            for params in paramsList:
                utilityTasks.append(LinRegTask(metadata, *params))

    ##################################
    ######### EVALUATION #############
    ##################################
    resultsTargetUtility = {ut.__name__: {gm.__name__: {} for gm in gmList + sanList} for ut in utilityTasks}
    resultsAggUtility = {ut.__name__: {gm.__name__: {'TargetID': [],
                                                     'Accuracy': []} for gm in gmList + sanList} for ut in utilityTasks}

    # Add entry for raw
    for ut in utilityTasks:
        resultsTargetUtility[ut.__name__]['Raw'] = {}
        resultsAggUtility[ut.__name__]['Raw'] = {'TargetID': [],
                                                 'Accuracy': []}

    print('\n---- Start the game ----')
    for nr in range(runconfig['nIter']):
        print(f'\n--- Game iteration {nr + 1} ---')
        # Draw a raw dataset
        rIdx = choice(list(rawTrainWoTargets.index), size=runconfig['sizeRawT'], replace=False).tolist()
        rawTout = rawTrain.loc[rIdx]

        LOGGER.info('Start: Utility evaluation on Raw...')
        # Get utility from raw without targets
        for ut in utilityTasks:
            resultsTargetUtility[ut.__name__]['Raw'][nr] = {}

            predErrorTargets = []
            predErrorAggr = []
            for _ in range(runconfig['nSynT']):
                ut.train(rawTout)
                predErrorTargets.append(ut.evaluate(testRecords))
                predErrorAggr.append(ut.evaluate(rawTest))

            resultsTargetUtility[ut.__name__]['Raw'][nr]['OUT'] = {
                'TestRecordID': testRecordIDs,
                'Accuracy': list(mean(predErrorTargets, axis=0))
            }

            resultsAggUtility[ut.__name__]['Raw']['TargetID'].append('OUT')
            resultsAggUtility[ut.__name__]['Raw']['Accuracy'].append(mean(predErrorAggr))

        # Get utility from raw with each target
        for tid in targetIDs:
            target = targets.loc[[tid]]
            rawIn = rawTout.append(target)

            for ut in utilityTasks:
                predErrorTargets = []
                predErrorAggr = []
                for _ in range(runconfig['nSynT']):
                    ut.train(rawIn)
                    predErrorTargets.append(ut.evaluate(testRecords))
                    predErrorAggr.append(ut.evaluate(rawTest))

                resultsTargetUtility[ut.__name__]['Raw'][nr][tid] = {
                    'TestRecordID': testRecordIDs,
                    'Accuracy': list(mean(predErrorTargets, axis=0))
                }

                resultsAggUtility[ut.__name__]['Raw']['TargetID'].append(tid)
                resultsAggUtility[ut.__name__]['Raw']['Accuracy'].append(mean(predErrorAggr))

        LOGGER.info('Finished: Utility evaluation on Raw.')

        for GenModel in gmList:
            LOGGER.info(f'Start: Evaluation for model {GenModel.__name__}...')
            GenModel.fit(rawTout)
            synTwithoutTarget = [GenModel.generate_samples(runconfig['sizeSynT']) for _ in range(runconfig['nSynT'])]
            synLabelsOut = [LABEL_OUT for _ in range(runconfig['nSynT'])]

            # Util evaluation for synthetic without all targets
            for ut in utilityTasks:
                resultsTargetUtility[ut.__name__][GenModel.__name__][nr] = {}

                predErrorTargets = []
                predErrorAggr = []
                for syn in synTwithoutTarget:
                    ut.train(syn)
                    predErrorTargets.append(ut.evaluate(testRecords))
                    predErrorAggr.append(ut.evaluate(rawTest))

                resultsTargetUtility[ut.__name__][GenModel.__name__][nr]['OUT'] = {
                    'TestRecordID': testRecordIDs,
                    'Accuracy': list(mean(predErrorTargets, axis=0))
                }

                resultsAggUtility[ut.__name__][GenModel.__name__]['TargetID'].append('OUT')
                resultsAggUtility[ut.__name__][GenModel.__name__]['Accuracy'].append(mean(predErrorAggr))


            for tid in targetIDs:
                LOGGER.info(f'Target: {tid}')
                target = targets.loc[[tid]]

                rawTin = rawTout.append(target)
                GenModel.fit(rawTin)
                synTwithTarget = [GenModel.generate_samples(runconfig['sizeSynT']) for _ in range(runconfig['nSynT'])]
                synLabelsIn = [LABEL_IN for _ in range(runconfig['nSynT'])]

                synT = synTwithoutTarget + synTwithTarget
                synTlabels = synLabelsOut + synLabelsIn

                # Util evaluation for synthetic with this target
                for ut in utilityTasks:
                    predErrorTargets = []
                    predErrorAggr = []
                    for syn in synTwithoutTarget:
                        ut.train(syn)
                        predErrorTargets.append(ut.evaluate(testRecords))
                        predErrorAggr.append(ut.evaluate(rawTest))

                    resultsTargetUtility[ut.__name__][GenModel.__name__][nr][tid] = {
                        'TestRecordID': testRecordIDs,
                        'Accuracy': list(mean(predErrorTargets, axis=0))
                    }

                    resultsAggUtility[ut.__name__][GenModel.__name__]['TargetID'].append(tid)
                    resultsAggUtility[ut.__name__][GenModel.__name__]['Accuracy'].append(mean(predErrorAggr))
            del synT, synTwithoutTarget, synTwithTarget

            LOGGER.info(f'Finished: Evaluation for model {GenModel.__name__}.')

        for San in sanList:
            LOGGER.info(f'Start: Evaluation for sanitiser {San.__name__}...')
            sanOut = San.sanitise(rawTout)

            for ut in utilityTasks:
                resultsTargetUtility[ut.__name__][San.__name__][nr] = {}

                predErrorTargets = []
                predErrorAggr = []
                for _ in range(runconfig['nSynT']):
                    ut.train(sanOut)
                    predErrorTargets.append(ut.evaluate(testRecords))
                    predErrorAggr.append(ut.evaluate(rawTest))

                resultsTargetUtility[ut.__name__][San.__name__][nr]['OUT'] = {
                    'TestRecordID': testRecordIDs,
                    'Accuracy': list(mean(predErrorTargets, axis=0))
                }

                resultsAggUtility[ut.__name__][San.__name__]['TargetID'].append('OUT')
                resultsAggUtility[ut.__name__][San.__name__]['Accuracy'].append(mean(predErrorAggr))

            for tid in targetIDs:
                LOGGER.info(f'Target: {tid}')
                target = targets.loc[[tid]]

                rawTin = rawTout.append(target)
                sanIn = San.sanitise(rawTin)

                sanT = [sanOut, sanIn]
                sanTLabels = [LABEL_OUT, LABEL_IN]


                for ut in utilityTasks:
                    predErrorTargets = []
                    predErrorAggr = []
                    for _ in range(runconfig['nSynT']):
                        ut.train(sanIn)
                        predErrorTargets.append(ut.evaluate(testRecords))
                        predErrorAggr.append(ut.evaluate(rawTest))

                    resultsTargetUtility[ut.__name__][San.__name__][nr][tid] = {
                        'TestRecordID': testRecordIDs,
                        'Accuracy': list(mean(predErrorTargets, axis=0))
                    }

                    resultsAggUtility[ut.__name__][San.__name__]['TargetID'].append(tid)
                    resultsAggUtility[ut.__name__][San.__name__]['Accuracy'].append(mean(predErrorAggr))

            del sanT, sanOut, sanIn

            LOGGER.info(f'Finished: Evaluation for model {San.__name__}.')

    outfile = f"ResultsUtilTargets_{dname}"
    LOGGER.info(f"Write results to {path.join(f'{args.outdir}', f'{outfile}')}")

    with open(path.join(f'{args.outdir}', f'{outfile}.json'), 'w') as f:
        json.dump(resultsTargetUtility, f, indent=2, default=json_numpy_serialzer)

    outfile = f"ResultsUtilAgg_{dname}"
    LOGGER.info(f"Write results to {path.join(f'{args.outdir}', f'{outfile}')}")

    with open(path.join(f'{args.outdir}', f'{outfile}.json'), 'w') as f:
        json.dump(resultsAggUtility, f, indent=2, default=json_numpy_serialzer)

if __name__ == "__main__":
    main()