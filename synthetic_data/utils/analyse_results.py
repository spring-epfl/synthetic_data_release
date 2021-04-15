import json
import matplotlib.pyplot as plt
import seaborn as sns

from numpy import mean
from pandas import DataFrame, to_numeric, concat
from itertools import cycle
from glob import glob
from os import path

from warnings import filterwarnings
filterwarnings('ignore')

from .evaluation_framework import get_record_privacy_gain, get_record_privacy_loss
from .plot_setup import set_style, colours as COLOURS, pltmarkers as MARKERS, fontsizelabels as FSIZELABELS, fontsizeticks as FSIZETICKS
set_style()

DATASETS = ['germancredit', 'adult', 'texas']
FEATURESET = ['Naive', 'Histogram', 'Correlations', 'Ensemble']
GMS = ['IndependentHistogram', 'BayesianNet', 'PrivBayesEps0.1']
DPGMS = ['IndependentHistogram', 'BayesianNet', 'PrivBayesEps1.6', 'PrivBayesEps0.1', 'PrivBayesEps0.05']
PATEGAN = ['PateGan', 'PateGanEps1', 'PateGanEps0.1', 'PateGanEps0.01']
ATTACKS = ['RandomForestClassifier', 'LogisticRegression', 'KNeighborsClassifier']

ORDERS = {'Dataset': DATASETS,
          'FeatureSet': FEATURESET,
          'GenerativeModel':GMS,
          'TargetModel': GMS,
          'AttackClassifier': ATTACKS}

FNAMES = {'Naive': '$\mathtt{F_{Naive}}$',
          'Histogram': '$\mathtt{F_{Hist}}$',
          'Correlations': '$\mathtt{F_{Corr}}$',
          'Ensemble': '$\mathtt{F_{Ens}}$'}

GMNAMES = {'IndependentHistogram': '$\mathtt{IndHist}$',
           # 'BayesianNet': '$\\varepsilon$ $\\rightarrow \infty$',
           'BayesianNet': '$\mathtt{BayNet}$',
           'CTGAN': '$\mathtt{CTGAN}$',
           'PrivBayesEps1.6': '$\mathtt{\\varepsilon: 1.6}$',
           'PrivBayesEps1': '$\mathtt{\\varepsilon: 1}$',
           'PrivBayesEps0.1': '$\mathtt{PrivBay}$',
           'PrivBayesEps0.05': '$\mathtt{\\varepsilon: 0.05}$',
           'PrivBayesEps0.01': '$\mathtt{\\varepsilon: 0.01}$',
           'PateGan': '$\\varepsilon$ $\\rightarrow \infty$',
           'PateGanEps1': '$\mathbf{\\varepsilon: 1}$',
           'PateGanEps0.1': '$\mathbf{\\varepsilon: 0.1}$',
           'PateGanEps0.01': '$\mathtt{PATEGAN}$'}

ACNAMES = {'RandomForestClassifier': '$\mathtt{RandForest}$',
           'LogisticRegression': '$\mathtt{LogReg}$',
           'KNeighborsClassifier': '$\mathtt{KNN}$'}

DNAMES = {'germancredit': '$\mathbf{Credit}$',
          'adult': '$\mathbf{Adult}$',
          'texas': '$\mathbf{Texas}$'}

NAMES = {'Dataset': DNAMES,
         'FeatureSet': FNAMES,
         'GenerativeModel':GMNAMES,
         'TargetModel':GMNAMES,
         'AttackClassifier': ACNAMES}

FEATURECMAP = {f:c for f,c in zip(FEATURESET, COLOURS[:len(FEATURESET)])}
GMCMAP = {g:c for g,c in zip(GMS, COLOURS[:len(GMS)])}
DPGMCMAP = {g:c for g,c in zip(DPGMS, COLOURS[:len(DPGMS)])}
ATTACKSCMAP = {a:c for a,c in zip(ATTACKS, COLOURS[:len(ATTACKS)])}

FEATUREMARKERS = {f:c for f,c in zip(FEATURESET, MARKERS[:len(FEATURESET)])}
GMMARKERS = {g:c for g,c in zip(GMS, MARKERS[:len(GMS)])}
DPGMMARKERS = {g:c for g,c in zip(DPGMS, MARKERS[:len(DPGMS)])}
ATTACKSMARKERS = {a:c for a,c in zip(ATTACKS, MARKERS[:len(ATTACKS)])}

MARKERCYCLE = cycle(MARKERS)


def load_results(directory, dname, attack='MIA'):
    """
    Load results of privacy evaluation
    :param directory: str: path/to/result/files
    :param dname: str: name of dataset for which to load results
    :return: DataFrame: results
    """
    files = glob(path.join(directory, f'{dname}*.json'))

    resList = []

    for fname in files:
        if attack == 'MIA':
            gm = fname.split('/')[-1].split(attack)[0].split(dname)[-1]

            if gm in GMS + DPGMS + PATEGAN:
                with open(fname) as file:
                    rd = json.load(file)

                rdf = parse_results_mia(rd)
                rdf['GenerativeModel'] = gm
                rdf['Dataset'] = dname
                resList.append(rdf)

        elif attack == 'MLE-AI':
            f = fname.split('/')[-1]
            gm = f.split(attack)[0].split(dname)[-1]
            sensitive = f.split(attack)[-1].split('.')[0]

            if gm in GMS + DPGMS + PATEGAN:
                with open(fname) as file:
                    rd = json.load(file)

                rdf = parse_results_attr_inf(rd)
                rdf['GenerativeModel'] = gm
                rdf['Dataset'] = dname
                rdf['SensitiveAttribute'] = sensitive
                resList.append(rdf)

        else:
            raise ValueError('Unknown attack type')

    return concat(resList)


def parse_results_mia(resDict):
    """ Parse results from privacy evaluation under MIA and aggregate by target and test run"""
    dfList = []

    for am, res in resDict.items():
        # Aggregate data by target and test run and average
        resDF = DataFrame(res).groupby(['TargetID', 'TestRun']).agg(mean).reset_index()

        # Get attack model details
        fset = [m for m in FEATURESET if m in am][0]
        amc = am.split(fset)[0]
        resDF['AttackClassifier'] = amc
        resDF['FeatureSet'] = fset

        dfList.append(resDF)

    results = concat(dfList)

    results['RecordPrivacyLossSyn'] = to_numeric(results['RecordPrivacyLossSyn'])
    results['RecordPrivacyLossRaw'] = to_numeric(results['RecordPrivacyLossRaw'])
    results['RecordPrivacyGain'] = to_numeric(results['RecordPrivacyGain'])
    results['ProbSuccess'] = to_numeric(results['ProbSuccess'])

    return results


def parse_results_attr_inf(resDict):
    """ Parse results of attribute inference privacy evaluation under attribute inference"""

    dfList = []

    for attack, res in resDict.items():
        # Aggregate data by target and test run and average
        resDF = DataFrame(res).groupby(['Target', 'TestRun']).agg(mean).reset_index()

        # Convert numerical values to correct dtype
        for c in list(resDF):
            if c != 'Target':
                if c != 'TestRun':
                    resDF[c] = to_numeric(resDF[c])

        resDF['RecordPrivacyLossRaw'] = list(map(lambda pR, pP: get_record_privacy_loss(pR, pP), resDF['ProbCorrectRawT'], resDF['ProbCorrectPrior']))
        resDF['RecordPrivacyLossSyn'] = list(map(lambda pR, pP: get_record_privacy_loss(pR, pP), resDF['ProbCorrectSynT'], resDF['ProbCorrectPrior']))
        resDF['RecordPrivacyGain'] = list(map(lambda pR, pS: get_record_privacy_gain(pR, pS), resDF['RecordPrivacyLossRaw'], resDF['RecordPrivacyLossSyn']))
        resDF['AttackModel'] = attack

        dfList.append(resDF)

    return concat(dfList)


def plt_summary(results, dname):
    """ Plot average privacy gain across all targets and datasets. """
    fig, ax = plt.subplots()
    pointplot(results, 'GenerativeModel', 'FeatureSet', ('Dataset', dname), ax)

    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.3), ncol=4, title='Feature Set')

    return fig

def plt_per_target(results, dname, fset='Naive'):
    """ Plot average privacy gain across all targets and datasets. """
    resultsF = results[results['FeatureSet'] == fset]

    fig, ax = plt.subplots()
    pointplot(resultsF, 'GenerativeModel', 'TargetID', ('FeatureSet', fset), ax)

    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.3), ncol=5, title='Target')

    return fig


def swarmplot(data, x, hue, filter, ax):
    sns.swarmplot(data=data, y='RecordPrivacyGain',
                  x=x, hue=hue,
                  order=ORDERS[x], hue_order=ORDERS[hue],
                  ax=ax, dodge=True)

    ax.set_title(f'{filter[0]}{NAMES[filter[0]][filter[1]]}', fontsize=FSIZELABELS)

    # Remove legend
    ax.get_legend().remove()

    # Set x- and y-label
    ax.set_xlabel('')
    ax.set_ylabel('$\mathtt{PG}_{\mathbf{t}}$', fontsize=FSIZELABELS)

    # Rename GMs
    ax.set_xticklabels([NAMES[x][xt.get_text()] for xt in ax.get_xticklabels()], fontsize=FSIZELABELS)

    # Resize y-tick labels
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(FSIZETICKS)

    ax.hlines(0.25, *ax.get_xlim(), 'grey', '--')


def pointplot(data, x, hue, filter, ax):
    ncats = data[hue].nunique()
    huemarkers = [next(MARKERCYCLE) for _ in range(ncats)]
    sns.pointplot(data=data, y='RecordPrivacyGain',
                  x=x, hue=hue,
                  order=ORDERS[x],# hue_order=ORDERS[hue],
                  ax=ax, dodge=True,
                  linestyles='', markers=huemarkers)

    ax.set_title(f'{filter[0]}: {NAMES[filter[0]][filter[1]]}', fontsize=FSIZELABELS)

    # Remove legend
    ax.get_legend().remove()

    # Set x- and y-label
    ax.set_xlabel('')
    ax.set_ylabel('$\mathtt{PG}_{\mathbf{t}}$', fontsize=FSIZELABELS)

    # Rename GMs
    ax.set_xticklabels([NAMES[x][xt.get_text()] for xt in ax.get_xticklabels()], fontsize=FSIZELABELS)

    # Resize y-tick labels
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(FSIZETICKS)

    ax.hlines(0.25, *ax.get_xlim(), 'grey', '--')