import json
import matplotlib.pyplot as plt
import seaborn as sns

from numpy import mean, ceil, arange, median, sqrt
from pandas import DataFrame, to_numeric, pivot_table, concat, read_csv
from itertools import combinations, combinations_with_replacement
from scipy import stats
from math import log10
from glob import glob
from os import path
from husl import hex_to_husl

from warnings import filterwarnings
filterwarnings('ignore')

from .evaluation_framework import get_record_privacy_gain, get_record_privacy_loss
from .plot_setup import set_style, cmap_diverging, cmap_light, colours as COLOURS, pltmarkers as MARKERS
set_style()

FEATURESET = ['Naive', 'Histogram', 'Correlations', 'Ensemble']
GMS = ['IndependentHistogram', 'BayesianNet', 'CTGAN']
DPGMS = ['BayesianNet', 'PrivBayesEps1', 'PrivBayesEps0.1', 'PrivBayesEps0.01']
PATEGAN = ['PateGan', 'PateGanEps1', 'PateGanEps0.1', 'PateGanEps0.01']
ATTACKS = ['RandomForestClassifier', 'LogisticRegression', 'KNeighborsClassifier']

FEATURECMAP = {f:c for f,c in zip(FEATURESET, COLOURS[:len(FEATURESET)])}
GMCMAP = {g:c for g,c in zip(GMS, COLOURS[:len(GMS)])}
DPGMCMAP = {g:c for g,c in zip(DPGMS, COLOURS[:len(DPGMS)])}
ATTACKSCMAP = {a:c for a,c in zip(ATTACKS, COLOURS[:len(ATTACKS)])}

FEATUREMARKERS = {f:c for f,c in zip(FEATURESET, MARKERS[:len(FEATURESET)])}
GMMARKERS = {g:c for g,c in zip(GMS, MARKERS[:len(GMS)])}
DPGMMARKERS = {g:c for g,c in zip(DPGMS, MARKERS[:len(DPGMS)])}
ATTACKSMARKERS = {a:c for a,c in zip(ATTACKS, MARKERS[:len(ATTACKS)])}


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


def load_datasets(resDir, gm):

    SA, labelsSA = [], []
    ST, labelsST = [], []

    files = glob(path.join(path.join(resDir, gm), 'SA_IN_*.csv'))
    for f in files:
        SA.append(read_csv(f, index_col=0))
        labelsSA.append(1)

    files = glob(path.join(path.join(resDir, gm), 'SA_OUT_*.csv'))
    for f in files:
        SA.append(read_csv(f, index_col=0))
        labelsSA.append(0)

    files = glob(path.join(path.join(resDir, gm), 'ST_IN_*.csv'))
    for f in files:
        ST.append(read_csv(f, index_col=0))
        labelsST.append(1)

    files = glob(path.join(path.join(resDir, gm), 'ST_OUT_*.csv'))
    for f in files:
        ST.append(read_csv(f, index_col=0))
        labelsST.append(0)

    return SA, labelsSA, ST, labelsST


def barplot(data, metric, hue, hue_order, ax=None):
    if ax:
        sns.barplot(x="Generative Model", y=metric, hue=hue, data=data,
                    order=GMS, hue_order=hue_order, ax=ax)
    else:
        return sns.barplot(x="Generative Model", y=metric, hue=hue, data=data,
                           order=GMS, hue_order=hue_order)


def boxplot(data, metric, hue, hue_order, ax=None):
    if ax:
        sns.boxplot(x="Generative Model", y=metric, hue=hue, data=data,
                    order=GMS, hue_order=hue_order, ax=ax)
    else:
        return sns.boxplot(x="Generative Model", y=metric, hue=hue, data=data,
                           order=GMS, hue_order=hue_order)


def stripplot(data, metric, hue, hue_order, ax=None):
    if ax:
        sns.stripplot(x="Generative Model", y=metric, hue=hue, data=data,
                    order=GMS, hue_order=hue_order, dodge=True, ax=ax)
    else:
        return sns.stripplot(x="Generative Model", y=metric, hue=hue, data=data,
                           order=GMS, dodge=True, hue_order=hue_order)


def swarmplot(data, metric, hue, hue_order, ax=None):
    if ax:
        sns.swarmplot(x="Generative Model", y=metric, hue=hue, data=data,
                      order=GMS, hue_order=hue_order, dodge=True, ax=ax)
    else:
        return sns.swarmplot(x="Generative Model", y=metric, hue=hue, data=data,
                             order=GMS, dodge=True, hue_order=hue_order)


def plt_compare_mia(data, metric, kind='swarm', models=GMS):
    g = sns.catplot(data=data, y=metric, kind=kind,
                    x='Generative Model', hue='Feature Set', col='Attack Classifier',
                    hue_order=FEATURESET, col_order=ATTACKS, order=models,
                    col_wrap=3, dodge=True, legend=False)
    g.fig.set_size_inches(16, 5)
    g.axes[0].legend(loc='upper center', bbox_to_anchor=(1.5, 1.4), ncol=4, title='Feature Set')

    for ax in g.axes.flat:
        # Draw baseline
        if int(log10(data[metric].max())) > 0:
            ax.hlines(50, *ax.get_xlim(), linestyle='--', colors='w')
        else:
            ax.hlines(.5, *ax.get_xlim(), linestyle='--', colors='w')

        # Make title more human-readable and larger
        if ax.get_title():
            ax.set_title(ax.get_title().split('=')[1],
                         fontsize='x-large')

        # Reduce text for xticklabels
        xticklabels = []
        for x in ax.get_xticklabels():
            if 'Independent' in x.get_text():
                xticklabels.append(x.get_text().split('Independent')[-1])
            elif 'Eps' in x.get_text():
                xticklabels.append(f"$\epsilon$: {x.get_text().split('Eps')[-1]}")
            else:
                xticklabels.append(x.get_text())
        ax.set_xticklabels(xticklabels, fontsize='large')

        # Increase fontsize for yticklabels
        if len(ax.get_yticklabels()) > 0:
            ax.set_yticklabels([y for y in ax.get_yticklabels()], fontsize='large')

        # Increase font size for x- and y-label
        ax.set_xlabel(ax.get_xlabel(), fontsize='x-large')
        ax.set_ylabel(ax.get_ylabel(), fontsize='x-large')

    return g.fig


def summarise_mean_performance(data, metric, attack='LogisticRegression', models=GMS):

    data = data[data['Attack Classifier'] == attack]

    idx = (models, FEATURESET)
    return pivot_table(data, values=metric, index='Generative Model', columns='Feature Set', aggfunc=mean).loc[idx]


def plt_mean_performance(data, metric, models=GMS, legend=False):
    g = sns.catplot(data=data, y=metric, kind='point',
                    x='Generative Model', hue='Feature Set', col='Attack Classifier',
                    hue_order=FEATURESET, col_order=ATTACKS, order=models,
                    linestyles='--', col_wrap=3, dodge=True, legend=False)

    g.fig.set_size_inches(16, 5)
    if legend:
        g.axes[0].legend(loc='upper center', bbox_to_anchor=(1.5, 1.4), ncol=4, title='Feature Set')

    for ax in g.axes.flat:
        # # Draw baseline
        # if int(log10(data[metric].max())) > 0:
        #     ax.hlines(50, *ax.get_xlim(), linestyle='--', colors='w')
        # else:
        #     ax.hlines(.5, *ax.get_xlim(), linestyle='--', colors='w')

        # Make title more human-readable and larger
        if ax.get_title():
            ax.set_title(ax.get_title().split('=')[1],
                         fontsize='x-large')

        # Reduce text for xticklabels
        xticklabels = []
        for x in ax.get_xticklabels():
            if 'Independent' in x.get_text():
                xticklabels.append(x.get_text().split('Independent')[-1])
            elif 'Eps' in x.get_text():
                xticklabels.append(f"$\epsilon$: {x.get_text().split('Eps')[-1]}")
            else:
                xticklabels.append(x.get_text())
        ax.set_xticklabels(xticklabels, fontsize='large')

        # Increase fontsize for yticklabels
        if len(ax.get_yticklabels()) > 0:
            ax.set_yticklabels([y for y in ax.get_yticklabels()], fontsize='large')

        # Increase font size for x- and y-label
        ax.set_xlabel(ax.get_xlabel(), fontsize='x-large')
        ax.set_ylabel(ax.get_ylabel(), fontsize='x-large')

    return g.fig



def plt_feature_comparison(data, metric, attack='LogisticRegression', models=GMS):

    res = summarise_ttest_stats(data, metric, attack, models)

    vmax = int(res.max().loc[slice(None),'$\Delta$'].max()+.6)
    vmin = int(res.min().loc[slice(None),'$\Delta$'].min()+.6)

    with sns.axes_style('white'):
        fig, ax = plt.subplots(1, len(models), figsize=(16,5), sharex=True, sharey=True)
        ax = ax.flatten()
        cbar_ax = fig.add_axes([.911, .15,.02,.7])

        for i, gm in enumerate(models):
            plt_data = res[gm]['$\Delta$'].to_frame().reset_index().pivot_table(index='Feature Set 1', columns='Feature Set 2')['$\Delta$']
            sns.heatmap(plt_data, cmap=cmap_diverging, vmin=vmin, vmax=vmax, center=0, square=True,annot=True, ax=ax[i], cbar_ax=cbar_ax)
            ax[i].set(title=gm)

        fig.suptitle(f'$\Delta$ Feature Set 1 $>$ Feature Set 2 for MIA {attack}', y=1.05)

    return fig


def summarise_ttest_stats(data, metric, attack='LogisticRegression', models=GMS):

    data = data[data['Attack Classifier'] == attack]

    gmgroups = data.groupby('Generative Model')

    gmresults = []
    for g in gmgroups.groups:
        df = gmgroups.get_group(g)
        fgroups = df.groupby('Feature Set')
        fcombs = [c for c in combinations(list(fgroups.groups), 2)]
        featureset1 = []
        featureset2 = []
        statistics = []
        pvalues = []

        for f in fcombs:
            t = ttest(fgroups, f[0], f[1], metric)
            featureset1.append(f[0])
            featureset2.append(f[1])
            statistics.append(t[0])
            pvalues.append(t[1])
        gms = [g for _ in range(len(fcombs))]

        r = DataFrame([featureset1, featureset2, statistics, pvalues, gms],
                      index=["Feature Set 1", 'Feature Set 2', '$\Delta$', 'p', 'Generative Model']).T
        r['$\Delta$'] = to_numeric(r['$\Delta$'])
        r['p'] = to_numeric(r['p'])

        gmresults.append(r)

    return pivot_table(concat(gmresults),
                       values=['$\Delta$', 'p'],
                       index=['Feature Set 1', 'Feature Set 2'],
                       columns='Generative Model').swaplevel(axis=1)[models]


def plt_cumulative(data, metric='Privacy Gain', attack='LogisticRegression', agg=median, cumulative=1, models=GMS):
    data = data[data['Attack Classifier'] == attack]
    results = data.groupby(['Target', 'Feature Set', 'Attack Classifier', 'Generative Model']).agg(agg).reset_index()
    nt = results['Target'].nunique()

    g = sns.FacetGrid(results, col="Generative Model", hue='Feature Set',
                      hue_order=FEATURECMAP, col_order=models)
    g.map(plt.hist, metric, cumulative=cumulative, histtype='step', bins=1000, linewidth=2)
    g.fig.subplots_adjust(wspace=.05, top=1)
    g.add_legend(bbox_to_anchor=(.4, 1.1), ncol=4, title='Feature Set', fancybox=True, frameon=True)
    g.fig.set_size_inches(25, 4)

    nrow, ncol = g.axes.shape

    for ai, ax in enumerate(g.axes.flat):
        # Make x and y-axis labels slightly larger
        ax.set_xlabel(ax.get_xlabel(), fontsize='large')

        if ai % ncol == 0:
            ax.set_ylabel('Number of targets', fontsize='large')
        else:
            ax.set_ylabel(ax.get_ylabel(), fontsize='large')
        ax.set_ylim([0, nt+1])

        # Make title more human-readable and larger
        if ax.get_title():
            l = ax.get_title().split('=')[1]
            if 'Independent' in l:
                l = l.split('Independent')[-1]
            elif 'Eps' in l:
                l = '$\epsilon$: %s'%(l.split('Eps')[-1])
            ax.set_title(l, fontsize='x-large')

        for patch in ax.patches:
            if cumulative == 1:
                patch.set_xy(patch.get_xy()[:-1])
            elif cumulative == -1:
                patch.set_xy(patch.get_xy()[1:])

    return g.fig


def plt_scatter(data, xmetric, ymetric, agg=median, hue='Feature Set', style='Generative Model', ax=None):
    results = data.groupby(['Target', 'Feature Set', 'Attack Classifier', 'Generative Model']).agg(agg).reset_index()


    if hue == 'Generative Model':
        hue_order = GMS
    elif hue == 'Attack Classifier':
        hue_order = ATTACKS
    elif hue == 'Feature Set':
        hue_order = FEATURESET
    else:
        raise ValueError(f'Unknown groupby {hue}')

    if style == 'Generative Model':
        markers = GMMARKERS
    elif style == 'Attack Classifier':
        markers = ATTACKSMARKERS
    elif style == 'Feature Set':
        markers = FEATUREMARKERS
    else:
        raise ValueError(f'Unknown groupby {hue}')

    if ax is not None:
        sns.scatterplot(x=xmetric, y=ymetric, hue=hue, style=style, data=results,
                        hue_order=hue_order, markers=markers, ax=ax)
    else:
        ax = sns.scatterplot(x=xmetric, y=ymetric, hue=hue, style=style, data=results,
                             hue_order=hue_order, markers=markers)

    ax.hlines([.25, .50, .75], *ax.get_xlim(), linestyle='--', colors='w')
    ax.vlines([.25, .50, .75], *ax.get_ylim(), linestyle='--', colors='w')
    ax.legend(loc='upper center', bbox_to_anchor=(1.2, 1))

    return ax


def plt_compare_features(data, metric, attack='LogisticRegression', kind='box'):
    assert attack in ATTACKS, f'Unkown attack model {attack}'

    data = data[data['Attack Classifier'] == attack]

    fig, ax = plt.subplots(figsize=(13, 6))
    if kind == 'bar':
        barplot(data, metric, 'Feature Set', FEATURESET, ax)
    elif kind == 'box':
        boxplot(data, metric, 'Feature Set', FEATURESET, ax)
    elif kind == 'strip':
        stripplot(data, metric, 'Feature Set', FEATURESET, ax)
    elif kind == 'swarm':
        swarmplot(data, metric, 'Feature Set', FEATURESET, ax)

    if int(log10(data[metric].max())) > 0:
        ax.hlines(50, *ax.get_xlim(), linestyle='--', colors='w')
        ax.set_ylim(-2,102)
    else:
        ax.hlines(.5, *ax.get_xlim(), linestyle='--', colors='w')
        ax.set_ylim(-0.01, 1.1)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=4, title='Feature Set')
    fig.suptitle(f'MIA Classifier {attack}', y=1.08)

    return fig


def plt_compare_posterior(data, models=GMS):
    sensgroups = data.groupby('Sensitive Attribute')

    fig, ax = plt.subplots(len(sensgroups), len(models), figsize=(len(models)*5, len(sensgroups)*5), sharey=True, sharex='row')
    if len(ax.shape) == 1:
        ax = ax.reshape(1, len(models))

    for j, (s, sdata) in enumerate(sensgroups):
        gmgroups = sdata.groupby('Generative Model')

        for i, gm in enumerate(models):

            gdata = gmgroups.get_group(gm)

            pdfCraftedRawT = stats.norm(loc=gdata['MLERawT'].mean(), scale=sqrt(gdata['SigmaRawT'].mean()))
            pdfCraftedSynT = stats.norm(loc=gdata['MLESynT'].mean(), scale=sqrt(gdata['SigmaSynT'].mean()))

            sns.distplot(pdfCraftedSynT.rvs(size=2000), label='$P[\hat t_s|S^t]$', ax=ax[j,i])
            ax[j,i].axvline(gdata['MLESynT'].mean(), ls='--', label='$E[\hat t_s | S^t]$')

            sns.distplot(pdfCraftedRawT.rvs(size=2000), label='$P[\hat t_s|R^t]$', ax=ax[j,i])
            ax[j,i].axvline(gdata['MLERawT'].mean(), ls='--', c=COLOURS[1], label='$E[\hat t_s | R^t]$')

            ax[j,i].axvline(gdata['TrueValue'].mean(), ls='--', color=COLOURS[2], label='True value')

            ax[j,i].set(title=gm, xlabel=s)

        ax[j, 0].set(ylabel='$P[\hat t_s]$')
        ax[0,0].legend(loc='upper center', bbox_to_anchor=(len(models)/2+.25, 1.32), ncol=5)

    fig.subplots_adjust(hspace=.4)

    return fig


def plt_avg_privgain(data, agg=median, models=GMS, markers=GMMARKERS):

    sensgroups = data.groupby('Sensitive Attribute')

    fig, ax = plt.subplots(len(sensgroups), 2, figsize=(15, len(sensgroups)*5))
    ax = ax.reshape(1, 2)

    for j, (s, sdata) in enumerate(sensgroups):
        sdata = sdata.groupby(['Target', 'Generative Model']).agg(agg).reset_index()
        sdata = sdata[sdata['Privacy Gain'] > sdata['Privacy Gain'].quantile(.05)]

        sns.catplot(x='Generative Model', y='Privacy Gain', hue='Generative Model',
                        data=sdata, order=models, hue_order=models, ax=ax[j, 0])
        plt.close()
        ax[j,0].set(title=s, ylabel='Avg Privacy Gain per Target')
        ax[j,0].legend().set_visible(False)

        sns.scatterplot(x='TrueValue', y='Privacy Gain', hue='Generative Model', style='Generative Model',
                        data=sdata, hue_order=models, markers=markers, ax=ax[j, 1])
        ax[j,1].set(title=s, xlabel='Target value', ylabel='Avg Privacy Gain per Target')

    fig.subplots_adjust(hspace=.3)

    return fig


def plt_compare_attacks(data, metric, featureset=None, kind='box'):

    if featureset is not None:
        assert featureset in FEATURESET, f'Unkown feature set {featureset}'
        data = data[data['Feature Set'] == featureset]

    fig, ax = plt.subplots(figsize=(13, 6))
    if kind == 'bar':
        barplot(data, metric, 'Attack Classifier', ATTACKS, ax)
    elif kind == 'box':
        boxplot(data, metric, 'Attack Classifier', ATTACKS, ax)
    elif kind == 'strip':
        stripplot(data, metric, 'Attack Classifier', ATTACKS, ax)
    elif kind == 'swarm':
        swarmplot(data, metric, 'Attack Classifier', ATTACKS, ax)

    if int(log10(data[metric].max())) > 0:
        ax.hlines(50, *ax.get_xlim(), linestyle='--', colors='w')
    else:
        ax.hlines(.5, *ax.get_xlim(), linestyle='--', colors='w')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, title='Attack Classifier')
    if featureset is not None:
        fig.suptitle(f'Feature Set {featureset}', y=1.05)

    return fig


def plt_compare_mean(data, metric, attack='LogisticRegression', ax=None):
    res = summarise_mean_performance(data, metric, attack)

    if ax is not None:
        sns.heatmap(res, vmin=0, vmax=100, square=True, annot=True, cmap=cmap_diverging, ax=ax)
    else:
        return sns.heatmap(res, vmin=0, vmax=100, square=True, annot=True, cmap=cmap_diverging, ax=ax)


def plt_matches(data, metric, between='Generative Model', by='Attack Classifier', cols='Feature Set', n=15):

    between_groups = list(data.groupby(between).groups.keys())
    combs = [c for c in combinations_with_replacement(between_groups, 2)]

    if cols == 'Generative Model':
        filtercmap = GMCMAP
    elif cols == 'Attack Classifier':
        filtercmap = ATTACKSCMAP
    elif cols == 'Feature Set':
        filtercmap = FEATURECMAP
    else:
        raise ValueError(f'Unknown groupby {between}')

    plots = []
    for fg, fgroup in data.groupby(cols):
        groups = fgroup.groupby(by)

        nrows = int(ceil(len(groups)/2))
        fig, ax = plt.subplots(nrows, 2, figsize=(10, nrows*4), sharex=True, sharey=True)
        ax = ax.flatten()

        for i, (g, group) in enumerate(groups):
            subgroups = group.groupby(between)

            pltdata = DataFrame(columns=between_groups, index=between_groups)
            prey = {}

            for sg, sgroup in subgroups:
                prey[sg] = set(sgroup.sort_values(metric)[['Target']].iloc[-n:].values.flatten())

            for c in combs:
                matches = len(prey[c[0]].intersection(prey[c[1]]))
                pltdata.loc[c[0], c[1]] = matches
                pltdata.loc[c[1], c[0]] = matches
            try:
                cmap = sns.light_palette(hex_to_husl(filtercmap[fg]), input="husl", as_cmap=True)
            except:
                cmap = cmap_light
            sns.heatmap(pltdata, ax=ax[i], square=True, annot=True, vmin=0, vmax=n, cmap=cmap)
            ax[i].set(title=g)
            ax[i].set_xticklabels([x for x in ax[i].get_xticklabels()], rotation=45, ha='right')

        if bool(len(groups)%2):
            ax[-1].axis('off')
        fig.suptitle(f'Matches between {n} most vulnerable targets by {by} for {cols} {fg}', y=1.05)
        plots.append(fig)

    return plots


def plt_type_error(data, metric, thresh=40, ax=None):

    untouchable = data[data[metric] <= thresh]

    if ax is not None:
        sns.scatterplot(x='FN', y='FP', size=metric, hue=metric, data=untouchable, palette='Blues', ax=ax)
    else:
        ax = sns.scatterplot(x='FN', y='FP', size=metric, hue=metric, data=untouchable, palette='Blues')
    ax.plot([i for i in range(50)], [i for i in range(50)], '--')
    ax.set(xlim=(0,51), ylim=(0,51))

    return ax


def ttest(groups, gm1, gm2, metric):
    df1 = groups.get_group(gm1)
    df2 = groups.get_group(gm2)

    tstats = stats.ttest_ind_from_stats(mean1=df1[metric].mean(), std1=df1[metric].std(), nobs1=len(df1),
                                        mean2=df2[metric].mean(), std2=df2[metric].std(), nobs2=len(df2))
    return tstats


def get_target_groups(data, low_thresh=25, up_thresh=75):

    nontargets = data[(data['Precision'] < 50) & (data['Accuracy'] < 50)]
    trickster = nontargets[(nontargets['Precision'] < low_thresh) & (nontargets['Accuracy'] < low_thresh)]
    repressed = nontargets[(nontargets['Precision'] < low_thresh) & (nontargets['Accuracy'] >= low_thresh)]
    fatamorgana = nontargets[(nontargets['Precision'] >= low_thresh) & (nontargets['Accuracy'] < low_thresh)]
    random = nontargets[(nontargets['Precision'] >= low_thresh) & (nontargets['Accuracy'] >= low_thresh)]

    prey = data[(data['Precision'] > up_thresh) & (data['Accuracy'] > up_thresh)]

    return nontargets, trickster, repressed, fatamorgana, random, prey


def compare_cohorts(R_df, targets, name, resultspath, dname):

    R_df['ID'] = [f'ID{i}' for i in arange(len(R_df))]

    cohorts = load_test_cohort(resultspath, dname)

    R_cohort = R_df.set_index('ID').loc[cohorts[dname]]
    R_cohort['Group'] = 'Target Test'
    R_attack = R_df.set_index('ID').drop(cohorts[dname])
    R_attack['Group'] = 'Attacker Train'

    targetid = targets['Target'].unique()
    R_targets = R_df.set_index('ID').loc[targetid]
    R_targets['Group'] = name

    R_plot = concat([R_attack, R_cohort, R_targets])
    catcols = list(R_plot.select_dtypes('object'))
    catcols.remove('Group')

    if len(list(R_plot)) - len(catcols) > 1:
        gpair = sns.pairplot(R_plot, hue='Group', markers=[".",  "o", "."], hue_order=['Attacker Train', name, 'Target Test'])
    else:
        gpair = None

    nrows = int(ceil(len(catcols)/3))
    fig, ax = plt.subplots(nrows, 3, figsize=(15, nrows*5))
    ax = ax.flatten()

    for i, col in enumerate(catcols):
        R_plot[col] = R_plot[col].apply(lambda x: ' '.join(x.split('_')))
        pltdata = R_plot.groupby(['Group', col])[col].size()/R_plot.groupby('Group').size() * 100
        sns.barplot(x=col, y=0, hue='Group', data=pltdata.to_frame().reset_index(), ax=ax[i], hue_order=['Attacker Train', name, 'Target Test'])
        ax[i].set(ylabel='\% Records in Group')
        if i > 0:
            ax[i].get_legend().remove()

    if len(catcols)%3 is not 0:
        for i in range(1,3):
            ax[-i].axis('off')

    return gpair, fig
