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

DATASETS = ['germancredit', 'adult', 'texas']
FEATURESET = ['Naive', 'Histogram', 'Correlations', 'Ensemble']
ATTACKS = ['RandomForestClassifier', 'LogisticRegression', 'KNeighborsClassifier']
PREDTASKS = ['RandomForestClassifier', 'LogisticRegression', 'LinearRegression']

GMS = ['IndependentHistogram', 'BayesianNet', 'CTGAN', 'PrivBayesEps0.1']
GMSPB = ['BayesianNet', 'PrivBayesEps1.0','PrivBayesEps0.1', 'PrivBayesEps0.05']
GMSDP = ['IndependentHistogram', 'BayesianNet', 'PrivBayesEps0.1', 'PateGanEps0.1']
GANS = ['CTGAN', 'PateGanEps1.6', 'PateGanEps0.1', 'PateGanEps0.05']
SAN = ['SanitiserNHSk5', 'SanitiserNHSk10', 'BayesianNet']

FNAMES = {'Naive': '$\mathtt{F_{Naive}}$',
          'Histogram': '$\mathtt{F_{Hist}}$',
          'Correlations': '$\mathtt{F_{Corr}}$',
          'Ensemble': '$\mathtt{F_{Ens}}$'}

GMNAMES = {'Raw': '$\mathtt{Raw}$',
           'IndependentHistogram': '$\mathtt{IndHist}$',
           'BayesianNet': '$\mathtt{BayNet}$',
           'CTGAN': '$\mathtt{CTGAN}$',
           #
           'PrivBayesEps1.6': '$\mathtt{PrivBay~\\varepsilon: 1.6}$',
           'PrivBayesEps1.0': '$\mathtt{PrivBay~\\varepsilon: 1}$',
           'PrivBayesEps0.1': '$\mathtt{PrivBay~\\varepsilon: 0.1}$',
           'PrivBayesEps0.05': '$\mathtt{PrivBay~\\varepsilon: 0.05}$',
           'PrivBayesEps0.01': '$\mathtt{PrivBay~\\varepsilon: 0.01}$',
           'PrivBayesEps1e-09': '$\mathtt{PrivBay~\\varepsilon \\rightarrow 0}$',
           #
           'PateGanEps1.6': '$\mathtt{PATEGAN~\\varepsilon: 1.6}$',
           'PateGanEps1.0': '$\mathtt{PATEGAN~\\varepsilon: 1}$',
           'PateGanEps0.1': '$\mathtt{PATEGAN~\\varepsilon: 0.1}$',
           'PateGanEps0.05': '$\mathtt{PATEGAN~\\varepsilon: 0.05}$',
           #
           'SanitiserNHSk2': '$\mathtt{San~k:2}$',
           'SanitiserNHSk5': '$\mathtt{San~k:5}$',
           'SanitiserNHSk10': '$\mathtt{San~k:10}$'
           }

DNAMES = {'germancredit': '$\mathbf{Credit}$',
          'adult': '$\mathbf{Adult}$',
          'texas': '$\mathbf{Texas}$',
          'texas_population': '$\mathbf{Texas}$'}

ATTRNAMES = {'TotalCharges': '\texttt{TotalCharges}',
             'LengthOfStay': '\\texttt{LengthOfStay}',
             'CapitalLoss': '$\mathbf{CapitalLoss}$',
             'RACE': '\\texttt{Race}'}

NAMES = {'Dataset': DNAMES,
         'FeatureSet': FNAMES,
         'GenerativeModel': GMNAMES,
         'TargetModel': GMNAMES,
         'SensitiveAttribute': ATTRNAMES}

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
    :param dpath: str: Dataset path (needed to extract some metadata)
    :return: results: DataFrame: Results of utility evaluation
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
def plt_summary(results, dname, models, hue='FeatureSet'):
    """ Plot average privacy gain across all targets and iterations. """
    fig, ax = plt.subplots()
    pointplot(results, 'TargetModel', 'PrivacyGain', hue, ax, models, ('Dataset', dname))

    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.3), ncol=4, title=hue)
    ax.set_ylabel('$\mathtt{PG}$', fontsize=FSIZELABELS)
    ax.set_ylim(-0.05)

    return fig


def plt_per_target(results, models, resFilter=('FeatureSet', 'Naive')):
    """ Plot per record average privacy gain. """
    results = results[results[resFilter[0]] == resFilter[1]]

    fig, ax = plt.subplots()
    pointplot(results, 'TargetModel', 'PrivacyGain', 'TargetID', ax, models, resFilter)

    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.3), ncol=5, title='TargetID')
    ax.set_ylabel('$\mathtt{PG}$', fontsize=FSIZELABELS)
    ax.set_ylim(-0.05)

    return fig


def plt_summary_accuracy(results, models):
    pltdata = results[results['TargetID'] == 'OUT']

    fig, ax = plt.subplots()
    boxplot(pltdata, 'TargetModel', 'Accuracy', 'LabelVar', ax, models)

    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.2), ncol=5, title='Target Attribute')
    ax.hlines(0.2, *ax.get_xlim(), 'grey', '--')
    ax.set_ylabel('Test Accuracy', fontsize=FSIZELABELS)

    return fig


def pointplot(data, x, y, hue, ax, order, filter=None):
    ncats = data[hue].nunique()
    huemarkers = HUEMARKERS[:ncats]

    sns.pointplot(data=data, y=y,
                  x=x, hue=hue,
                  order=order, ci='sd',
                  ax=ax, dodge=True,
                  join=False, markers=huemarkers,
                  scale=1.2, errwidth=1)

    if filter is not None:
        ax.set_title(f'{filter[0]}: {NAMES[filter[0]][filter[1]]}', fontsize=FSIZELABELS)

    # Remove legend
    ax.get_legend().remove()

    # Set x- and y-label
    ax.set_xlabel('')

    # Rename GMs
    ax.set_xticklabels([NAMES[x][xt.get_text()] for xt in ax.get_xticklabels()], fontsize=FSIZELABELS)

    # Resize y-tick labels
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(FSIZETICKS)


def boxplot(data, x, y, hue, ax, order, filter=None, hue_order=None):
    sns.boxenplot(data=data, y=y,
                  x=x, hue=hue,
                  order=order, hue_order=hue_order,
                  ax=ax, dodge=True)

    if filter is not None:
        ax.set_title(f'{filter[0]}: {NAMES[filter[0]][filter[1]]}', fontsize=FSIZELABELS)

    # Remove legend
    ax.get_legend().remove()

    # Set x- and y-label
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Rename GMs
    ax.set_xticklabels([NAMES[x][xt.get_text()] for xt in ax.get_xticklabels()], fontsize=FSIZELABELS)

    # Resize y-tick labels
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(FSIZETICKS)
