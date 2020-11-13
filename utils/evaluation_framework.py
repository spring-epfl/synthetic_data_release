"""
Procedures for running a privacy evaluation on a generative model
"""

from sklearn.metrics import roc_curve, auc
from os import path
from numpy import concatenate, mean, ndarray
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
from multiprocessing import Pool

from privacy_attacks.membership_inference import LABEL_IN, LABEL_OUT, generate_mia_shadow_data_shufflesplit
from privacy_attacks.attribute_inference import AttributeInferenceAttackLinearRegression
from utils.datagen import convert_df_to_array

from warnings import filterwarnings
filterwarnings('ignore')

from logging import getLogger
from logging.config import fileConfig

cwd = path.dirname(__file__)

logconfig = path.join(cwd, '../logging_config.ini')
fileConfig(logconfig)
logger = getLogger()

PROCESSES = 16


def get_roc_auc(trueLables, scores):
    """
    Calculate ROC curve of a binary classifier
    :param trueLables: list: ground truth
    :param scores: list: scores of a binary classifier for correct class
    :return: tuple: (false positive rate, true positive rate, auc)
    """
    fpr, tpr, _ = roc_curve(trueLables, scores)
    area = auc(fpr, tpr)

    return fpr, tpr, area


def get_scores(classProbabilities, trueLabels):
    """
    Get scores of correct class
    :param classProbabilities: list: list of arrays of prediction probabilities for each class
    :param trueLabels: list: correct class labels
    :return: list: scores for correct class
    """
    return [p[l] for p, l in zip(classProbabilities, trueLabels)]


def get_fp_tn_fn_tp(guesses, trueLabels):
    fp = sum([g == LABEL_IN for g,s in zip(guesses, trueLabels) if s == LABEL_OUT])
    tn = sum([g == LABEL_OUT for g,s in zip(guesses, trueLabels) if s == LABEL_OUT])
    fn = sum([g == LABEL_OUT for g,s in zip(guesses, trueLabels) if s == LABEL_IN])
    tp = sum([g == LABEL_IN for g,s in zip(guesses, trueLabels) if s == LABEL_IN])

    return fp, tn, fn, tp


def get_attack_accuracy(tp, tn, nguesses):
    return (tp + tn)/nguesses


def get_attack_precision(fp, tp):
    try:
        return tp / (tp + fp)
    except ZeroDivisionError as e:
        return .5


def get_attack_recall(tp, fn):
    try:
        return tp / (tp + fn)
    except ZeroDivisionError as e:
        return .5


def get_record_privacy_loss(pSuccess, pPrior):
    return pSuccess - pPrior


def get_record_privacy_gain(privLossRawT, privLossSynT):
        return (privLossRawT - privLossSynT) / 2


def evaluate_mia(GenModel, attacksList, rawWithoutTargets, targetRecords, targetIDs, rawAidx, rawTindices, sizeRawT, sizeSynT, nSynT, nSynA, nShadows, metadata):

    logger.info(f'Start evaluation of generative target model {GenModel.__name__} on {len(targetIDs)} targets under {len(attacksList)} different MIA models.')

    # Train and test MIA per target
    with Pool(processes=PROCESSES) as pool:
        tasks = [(GenModel, attacksList, targetRecords.loc[[tid], :], tid, rawWithoutTargets, metadata, rawAidx, rawTindices, sizeRawT, sizeSynT, nSynT, nSynA, nShadows) for tid in targetIDs]
        resultsList = pool.map(worker_run_mia, tasks)

    results = {
        AM.__name__: {
            'TargetID': [],
            'TestRun': [],
            'ProbSuccess': [],
            'RecordPrivacyLossSyn': [],
            'RecordPrivacyLossRaw': [],
            'RecordPrivacyGain': []
        } for AM in attacksList }

    for res in resultsList:
        for AM in attacksList:
            for k, v in res[AM.__name__].items():
                results[AM.__name__][k].extend(res[AM.__name__][k])

    for AM in attacksList:
        logger.info(f'Mean record privacy gain across {len(targetRecords)} Targets with Attack {AM.__name__}: {mean(results[AM.__name__]["RecordPrivacyGain"])}%')

    return results


def worker_run_mia(params):

    GenModel, attacksList, target, targetID, rawWithoutTargets, metadata, rawAidx, rawTindices, sizeRawT, sizeSynT, nSynT, nSynA, nShadows = params

    # Generate shadow model data for training attacks on this target
    if GenModel.datatype is DataFrame:
        rawA = rawWithoutTargets.loc[rawAidx, :]
    else:
        rawA = convert_df_to_array(rawWithoutTargets.loc[rawAidx, :], metadata)
        target = convert_df_to_array(target, metadata)

    synA, labelsSynA = generate_mia_shadow_data_shufflesplit(GenModel, target, rawA, sizeRawT, sizeSynT, nShadows, nSynA)

    for Attack in attacksList:
        Attack.train(synA, labelsSynA)

    # Clean up
    del synA, labelsSynA

    results = {
        AM.__name__: {
            'TargetID': [],
            'TestRun': [],
            'ProbSuccess': [],
            'RecordPrivacyLossSyn': [],
            'RecordPrivacyLossRaw': [],
            'RecordPrivacyGain': []
        } for AM in attacksList }

    for nr, rt in enumerate(rawTindices):

        # Generate synthetic datasets from generative model trained on RawT WITHOUT Target
        if GenModel.datatype is DataFrame:
            rawTout = rawWithoutTargets.loc[rt, :]
        else:
            rawTout = convert_df_to_array(rawWithoutTargets.loc[rt, :], metadata)

        GenModel.fit(rawTout) # Fit model
        synTwithoutTarget = [GenModel.generate_samples(sizeSynT) for _ in range(nSynT)]

        # Generate synthetic datasets from generative model trained on RawT PLUS Target
        if GenModel.datatype is DataFrame:
            rawTin = rawTout.append(target)
        else:
            if len(target.shape) == 1:
                target = target.reshape(1, len(target))
            rawTin = concatenate([rawTout, target])

        GenModel.fit(rawTin)
        synTwithTarget = [GenModel.generate_samples(sizeSynT) for _ in range(nSynT)]

        # Create balanced test dataset
        synT = synTwithTarget + synTwithoutTarget
        labelsSynT = [LABEL_IN for _ in range(len(synTwithTarget))] + [LABEL_OUT for _ in range(len(synTwithoutTarget))]

        # Run attacks on synthetic datasets from target generative model
        for AM in attacksList:
            res = run_mia(AM, synT, labelsSynT, targetID, nr)
            for k, v in res.items():
                results[AM.__name__][k].extend(v)

        del synT, labelsSynT

    return results


def run_mia(Attack, synT, labelsSynT, targetID, nr):

    probSuccessSyn = Attack.get_probability_of_success(synT, labelsSynT)
    priorProb = [Attack._get_prior_probability(s) for s in labelsSynT]
    privLossSyn = [get_record_privacy_loss(ps, pp) for ps, pp in zip(probSuccessSyn, priorProb)]
    privLossRaw = [get_record_privacy_loss(1, pp) for pp in priorProb]
    privGain = [get_record_privacy_gain(plR, plS) for plR, plS in zip(privLossRaw, privLossSyn)]

    results = {
        'TargetID': [targetID for _ in labelsSynT],
        'TestRun': [f'Run {nr + 1}' for _ in labelsSynT],
        'ProbSuccess': probSuccessSyn,
        'RecordPrivacyLossSyn': privLossSyn,
        'RecordPrivacyLossRaw': privLossRaw,
        'RecordPrivacyGain': privGain
    }

    return results


def craft_outlier(data, size):

    # Craft outlier target
    outlier = DataFrame(columns=list(data))
    for attr in list(data):
        if is_numeric_dtype(data[attr]):
            outlier[attr] = [data[attr].max() for _ in range(size)]
        else:
            outlier[attr] = [data[attr].value_counts().index[-1] for _ in range(size)]
    outlier.index = ['Crafted' for _ in range(size)]

    return outlier


def evaluate_ai(GenModel, rawWithoutTargets, targetRecords, targetIDs, rawA, rawTindices, sensitiveAttribute, sizeSynT, nSynT, metadata):

    logger.info(f'Start evaluation of generative target model {GenModel.__name__} on {len(targetIDs)} targets under MLE-AI.')

    results = {'LinearRegression':
        {
            'Target': [],
            'TrueValue': [],
            'ProbCorrectPrior': [],
            'MLERawT': [],
            'SigmaRawT': [],
            'ProbCorrectRawT': [],
            'MLESynT': [],
            'SigmaSynT': [],
            'ProbCorrectSynT': [],
            'TestRun': []
        }
    }

    for nr, rt in enumerate(rawTindices):
        logger.info(f'Raw target test set {nr+1}/{len(rawTindices)}')

        # Get raw target test set
        rawT = rawWithoutTargets.loc[rt, :]

        # Get baseline from raw data
        AttackBaseline = AttributeInferenceAttackLinearRegression(sensitiveAttribute, metadata, rawA)
        logger.info(f'Train Attack {AttackBaseline.__name__} on RawT')
        AttackBaseline.train(rawT)

        # Train generative model on raw data and sample synthetic copies
        logger.info(f'Start fitting {GenModel.__class__.__name__} to RawT')
        if GenModel.datatype is ndarray:
            rawT = convert_df_to_array(rawT, metadata)

        GenModel.fit(rawT)
        logger.info(f'Sample {nSynT} copies of synthetic data from {GenModel.__class__.__name__}')
        synT = [GenModel.generate_samples(sizeSynT) for _ in range(nSynT)]

        logger.info(f'Start Attack evaluation on SynT for {len(targetIDs)} targets')

        with Pool(processes=PROCESSES) as pool:
            tasks = [(s, targetRecords, targetIDs, sensitiveAttribute, AttackBaseline, metadata, rawA) for s in synT]
            resList = pool.map(worker_run_mleai, tasks)

        # Gather results
        for res in resList:
            for k, v in res[AttackBaseline.__name__].items():
                results[AttackBaseline.__name__][k].extend(v)
            results[AttackBaseline.__name__]['TestRun'].extend([f'Run {nr + 1}' for _ in range(len(targetIDs))])

    return results


def worker_run_mleai(params):
    """Worker for parallel processing"""
    syn, targetRecords, targetIDs, sensitiveAttribute, AttackBaseline, metadata, rawA = params

    results = {AttackBaseline.__name__:
                {
                    'Target': [],
                    'TrueValue': [],
                    'ProbCorrectPrior': [],
                    'MLERawT': [],
                    'SigmaRawT': [],
                    'ProbCorrectRawT': [],
                    'MLESynT': [],
                    'SigmaSynT': [],
                    'ProbCorrectSynT': []
                }
    }


    Attack = AttributeInferenceAttackLinearRegression(sensitiveAttribute, metadata, rawA)
    Attack.train(syn)

    for tid in targetIDs:
        t = targetRecords.loc[[tid], :]
        targetAux = t.drop_duplicates().drop(sensitiveAttribute, axis=1)
        targetSecret = t.drop_duplicates().loc[tid, sensitiveAttribute]

        # Baseline on raw data
        results[AttackBaseline.__name__]['Target'].append(tid)
        results[AttackBaseline.__name__]['TrueValue'].append(targetSecret)
        results[AttackBaseline.__name__]['ProbCorrectPrior'].append(AttackBaseline.get_prior_probability(targetSecret))
        results[AttackBaseline.__name__]['SigmaRawT'].append(AttackBaseline.sigma)
        results[AttackBaseline.__name__]['MLERawT'].extend(AttackBaseline.attack(targetAux).tolist())
        results[AttackBaseline.__name__]['ProbCorrectRawT'].extend(AttackBaseline.get_likelihood(targetAux, targetSecret).tolist())

        # Get attack results for this synthetic dataset
        results[Attack.__name__]['SigmaSynT'].append(Attack.sigma)
        results[Attack.__name__]['MLESynT'].extend(Attack.attack(targetAux).tolist())
        results[Attack.__name__]['ProbCorrectSynT'].extend(Attack.get_likelihood(targetAux, targetSecret).tolist())

    return results

