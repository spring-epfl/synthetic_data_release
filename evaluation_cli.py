"""
Command line interface to evaluate the trade-off between privacy gain and utility loss in a synthetic dataset.
"""

from argparse import ArgumentParser
from pathlib import Path
import pickle

import json

from os import mkdir, path
from numpy import mean
from numpy.random import choice
from argparse import ArgumentParser

from utils.datagen import load_s3_data_as_df, load_local_data_as_df
from utils.utils import json_numpy_serialzer
from utils.logging import LOGGER

from predictive_models.predictive_model import RandForestClassTask, LogRegClassTask, LinRegTask

from warnings import simplefilter

simplefilter('ignore', category=FutureWarning)
simplefilter('ignore', category=DeprecationWarning)

cwd = path.dirname(__file__)


def main():
    argparser = ArgumentParser()
    datasource = argparser.add_mutually_exclusive_group()
    datasource.add_argument(
        "--s3name",
        "-S3",
        type=str,
        choices=["adult", "census", "credit", "alarm", "insurance"],
        help="Name of the dataset to run on"
    )
    datasource.add_argument(
        "--datapath",
        "-D",
        type=Path,
        help="Relative path to cwd of a local data file"
    )
    argparser.add_argument(
        "--model",
        "-m",
        type=Path,
        help="Model"
    )
    argparser.add_argument(
        "--runconfig",
        "-RC",
        default="runconfig_mia.json",
        type=Path,
        help="Path relative to cwd of runconfig file"
    )
    argparser.add_argument(
        "--outdir",
        "-O",
        default="outputs/test",
        type=Path,
        help="Path relative to cwd for storing output files"
    )
    args = argparser.parse_args()

    with args.model.open('br') as fd:
        model = pickle.load(fd)

    # Load runconfig
    with args.runconfig.open() as f:
        runconfig = json.load(f)
    print('Runconfig:')
    print(runconfig)

    # Load data
    if args.s3name is not None:
        rawPop, metadata = load_s3_data_as_df(args.s3name)
        dname = args.s3name
    else:
        rawPop, metadata = load_local_data_as_df(path.join(cwd, args.datapath))
        dname = args.datapath.stem

    print(f'Loaded data {dname}:')
    print(rawPop.info())

    # Make sure outdir exists
    if not path.isdir(args.outdir):
        mkdir(args.outdir)

    ########################
    #### GAME INPUTS #######
    ########################
    # Train test split
    rawTrain = rawPop.query(runconfig['dataFilter']['train'])
    rawTest = rawPop.query(runconfig['dataFilter']['test'])

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
    resultsTargetUtility = {ut.__name__: {model.__name__: {}} for ut in utilityTasks}
    resultsAggUtility = {
        ut.__name__: {
            model.__name__: {
                'TargetID': [],
                'Accuracy': []
            }
        } for ut in utilityTasks
    }

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

        LOGGER.info(f'Start: Evaluation for model {model.__name__}...')
        model.fit(rawTout)
        synTwithoutTarget = [model.generate_samples(runconfig['sizeSynT']) for _ in range(runconfig['nSynT'])]

        # Util evaluation for synthetic without all targets
        for ut in utilityTasks:
            resultsTargetUtility[ut.__name__][model.__name__][nr] = {}

            predErrorTargets = []
            predErrorAggr = []
            for syn in synTwithoutTarget:
                ut.train(syn)
                predErrorTargets.append(ut.evaluate(testRecords))
                predErrorAggr.append(ut.evaluate(rawTest))

            resultsTargetUtility[ut.__name__][model.__name__][nr]['OUT'] = {
                'TestRecordID': testRecordIDs,
                'Accuracy': list(mean(predErrorTargets, axis=0))
            }

            resultsAggUtility[ut.__name__][model.__name__]['TargetID'].append('OUT')
            resultsAggUtility[ut.__name__][model.__name__]['Accuracy'].append(mean(predErrorAggr))

        for tid in targetIDs:
            LOGGER.info(f'Target: {tid}')
            target = targets.loc[[tid]]

            rawTin = rawTout.append(target)
            model.fit(rawTin)
            synTwithTarget = [model.generate_samples(runconfig['sizeSynT']) for _ in range(runconfig['nSynT'])]

            # Util evaluation for synthetic with this target
            for ut in utilityTasks:
                predErrorTargets = []
                predErrorAggr = []
                for syn in synTwithTarget:
                    ut.train(syn)
                    predErrorTargets.append(ut.evaluate(testRecords))
                    predErrorAggr.append(ut.evaluate(rawTest))

                resultsTargetUtility[ut.__name__][model.__name__][nr][tid] = {
                    'TestRecordID': testRecordIDs,
                    'Accuracy': list(mean(predErrorTargets, axis=0))
                }

                resultsAggUtility[ut.__name__][model.__name__]['TargetID'].append(tid)
                resultsAggUtility[ut.__name__][model.__name__]['Accuracy'].append(mean(predErrorAggr))

        del synTwithoutTarget, synTwithTarget

        LOGGER.info(f'Finished: Evaluation for model {model.__name__}.')

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
