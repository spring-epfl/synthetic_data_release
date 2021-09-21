import json
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
from pandas import DataFrame, concat
from itertools import cycle
from os import path

from warnings import filterwarnings
filterwarnings('ignore')

from .datagen import load_local_data_as_df
from .plot_setup import set_style, pltmarkers as MARKERS, fontsizelabels as FSIZELABELS, fontsizeticks as FSIZETICKS
from .evaluation_framework import *
set_style()

PREDTASKS = ['RandomForestClassifier', 'LogisticRegression', 'LinearRegression']

MARKERCYCLE = cycle(MARKERS)
HUEMARKERS = [next(MARKERCYCLE) for _ in range(20)]


###### Load results
def load_results_linkage(dirname):
    """
    Helper function to load results of privacy evaluation under risk of linkability
    :param dirname: str: Directory that contains results files
    :return: results: DataFrame: Results of privacy evaluation
    """

    files = glob(path.join(dirname, f'ResultsMIA_*.json'))

    resList = []
    for fpath in files:
        with open(fpath) as f:
            resDict = json.load(f)

        dataset = fpath.split('.json')[0].split('_')[-1]

        for tid, tres in resDict.items():
            for gm, gmDict in tres.items():
                for nr, nrDict in gmDict.items():
                    for fset, fsetDict in nrDict.items():
                        df = DataFrame(fsetDict)

                        df['Run'] = nr
                        df['FeatureSet'] = fset
                        df['TargetModel'] = gm
                        df['TargetID'] = tid
                        df['Dataset'] = dataset

                        resList.append(df)

    results = concat(resList)

    resAgg = []

    games = results.groupby(['TargetID', 'TargetModel', 'FeatureSet', 'Run'])
    for gameParams, gameRes in games:
        tpSyn, fpSyn = get_tp_fp_rates(gameRes['AttackerGuess'], gameRes['Secret'])
        advantageSyn = get_mia_advantage(tpSyn, fpSyn)
        advantageRaw = 1

        resAgg.append(gameParams + (tpSyn, fpSyn, advantageSyn, advantageRaw))

    resAgg = DataFrame(resAgg)

    resAgg.columns = ['TargetID','TargetModel', 'FeatureSet', 'Run', 'TPSyn', 'FPSyn', 'AdvantageSyn', 'AdvantageRaw']

    resAgg['PrivacyGain'] = resAgg['AdvantageRaw'] - resAgg['AdvantageSyn']

    return resAgg


def load_results_inference(dirname, dpath):
    """
    Helper function to load results of privacy evaluation under risk of inference
    :param dirname: str: Directory that contains results files
    :param dpath: str: Dataset path (needed to extract some metadata)
    :return: results: DataFrame: Results of privacy evaluation
    """
    df, metadata = load_local_data_as_df(dpath)

    files = glob(path.join(dirname, f'ResultsMLEAI_*.json'))
    resList = []
    for fpath in files:

        with open(fpath) as f:
            resDict = json.load(f)

        dataset = fpath.split('.json')[0].split('_')[-1]

        for tid, tdict in resDict.items():
            for sa, sdict in tdict.items():
                tsecret = df.loc[tid, sa]
                satype = None

                for cdict in metadata['columns']:
                    if cdict['name'] == sa:
                        satype = cdict['type']

                if '_' in sa:
                    sa = ''.join([s.capitalize() for s in sa.split('_')])
                elif '-' in sa:
                    sa = ''.join([s.capitalize() for s in sa.split('-')])

                for gm, gdict in sdict.items():
                    for nr, res in gdict.items():

                        resDF = DataFrame(res)
                        resDF['TargetID'] = tid
                        resDF['TargetSecret'] = tsecret
                        resDF['SensitiveType'] = satype
                        resDF['TargetModel'] = gm
                        resDF['Run'] = nr
                        resDF['SensitiveAttribute'] = sa
                        resDF['Dataset'] = dataset

                        resList.append(resDF)

    results = concat(resList)

    resAdv = []
    for gameParams, game in results.groupby(['Dataset', 'TargetID', 'SensitiveAttribute', 'Run']):
        rawRes = game.groupby(['TargetModel']).get_group('Raw')
        if all(game['SensitiveType'].isin([INTEGER, FLOAT])):
            pCorrectRIn, pCorrectROut = get_probs_correct(rawRes['ProbCorrect'], rawRes['TargetPresence'])

        elif all(game['SensitiveType'].isin([CATEGORICAL, ORDINAL])):
            pCorrectRIn, pCorrectROut = get_accuracy(rawRes['AttackerGuess'], rawRes['TargetSecret'], rawRes['TargetPresence'])

        else:
            raise ValueError('Unknown sensitive attribute type.')

        advR = get_ai_advantage(pCorrectRIn, pCorrectROut)

        for gm, gmRes in game.groupby(['TargetModel']):
            if gm != 'Raw':
                if all(gmRes['SensitiveType'].isin([INTEGER, FLOAT])):
                    pCorrectSIn, pCorrectSOut = get_probs_correct(gmRes['ProbCorrect'], gmRes['TargetPresence'])

                elif all(gmRes['SensitiveType'].isin([CATEGORICAL, ORDINAL])):
                    pCorrectSIn, pCorrectSOut = get_accuracy(gmRes['AttackerGuess'], gmRes['TargetSecret'], gmRes['TargetPresence'])

                else:
                    raise ValueError('Unknown sensitive attribute type.')

                advS = get_ai_advantage(pCorrectSIn, pCorrectSOut)


                resAdv.append(gameParams + (gm, pCorrectRIn, pCorrectROut, advR, pCorrectSIn, pCorrectSOut, advS))


    resAdv = DataFrame(resAdv)
    resAdv.columns  =['Dataset', 'TargetID', 'SensitiveAttribute','Run', 'TargetModel',
                      'ProbCorrectRawIn', 'ProbCorrectRawOut', 'AdvantageRaw',
                      'ProbCorrectSynIn', 'ProbCorrectSynOut', 'AdvantageSyn']

    resAdv['PrivacyGain'] = resAdv['AdvantageRaw'] - resAdv['AdvantageSyn']

    return resAdv


def load_results_utility(dirname):
    """
    Helper function to load results of utility evaluation
    :param dirname: str: Directory that contains results files
    :return: resultsTarget: DataFrame: Results of utility evaluation on individual records
    :return: resultsAgg: DataFrame: Results of average utility evaluation
    """

    # Load individual target utility results
    files = glob(path.join(dirname, f'ResultsUtilTargets_*.json'))

    resList = []
    for fpath in files:
        with open(fpath) as f:
            results = json.load(f)

        dataset = fpath.split('.json')[0].split('_')[-1]

        for ut, ures in results.items():
            model = [m for m in PREDTASKS if m in ut][0]
            labelVar = ut.split(model)[-1]

            if '_' in labelVar:
                labelVar = ''.join([s.capitalize() for s in labelVar.split('_')])

            if '-' in labelVar:
                labelVar = ''.join([s.capitalize() for s in labelVar.split('-')])

            for gm, gmres in ures.items():
                for n, nres in gmres.items():
                    for tid, tres in nres.items():
                        res = DataFrame(tres)

                        res['TargetID'] = tid
                        res['Run'] = f'Run {n}'
                        res['TargetModel'] = gm
                        res['PredictionModel'] = model
                        res['LabelVar'] = labelVar
                        res['Dataset'] = dataset

                        resList.append(res)

    resultsTargets = concat(resList)

    # Load aggregate utility results
    files = glob(path.join(dirname, f'ResultsUtilAgg_*.json'))

    resList = []
    for fpath in files:
        with open(fpath) as f:
            results = json.load(f)

        dataset = fpath.split('.json')[0].split('_')[-1]

        for ut, utres in results.items():
            model = [m for m in PREDTASKS if m in ut][0]
            labelVar = ut.split(model)[-1]

            if '_' in labelVar:
                labelVar = ''.join([s.capitalize() for s in labelVar.split('_')])

            if '-' in labelVar:
                labelVar = ''.join([s.capitalize() for s in labelVar.split('-')])

            for gm, gmres in utres.items():
                resDF = DataFrame(gmres)
                resDF['PredictionModel'] = model
                resDF['LabelVar'] = labelVar
                resDF['TargetModel'] = gm
                resDF['Dataset'] = dataset

                resList.append(resDF)

    resultsAgg = concat(resList)

    return resultsTargets, resultsAgg


### Plotting
def plt_per_target_pg(results, models, resFilter=('FeatureSet', 'Naive')):
    """ Plot per record average privacy gain. """
    results = results[results[resFilter[0]] == resFilter[1]]

    fig, ax = plt.subplots()
    pointplot(results, 'TargetModel', 'PrivacyGain', 'TargetID', ax, models)

    ax.set_title()
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.3), ncol=5, title='TargetID')
    ax.set_ylabel('$\mathtt{PG}$', fontsize=FSIZELABELS)
    ax.set_ylim(-0.05)

    return fig


def pointplot(data, x, y, hue, ax, order):
    ncats = data[hue].nunique()
    huemarkers = HUEMARKERS[:ncats]

    sns.pointplot(data=data, y=y,
                  x=x, hue=hue,
                  order=order, ci='sd',
                  ax=ax, dodge=True,
                  join=True, markers=huemarkers,
                  scale=1.2, errwidth=2,
                  linestyles='--')

    # Remove legend
    ax.get_legend().remove()

    # Set x- and y-label
    ax.set_xlabel('')

    # Resize y-tick labels
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(FSIZETICKS)
