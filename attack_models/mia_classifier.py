"""Parent class for launching a membership inference attack on the output of a generative model"""
from pandas import DataFrame
from pandas.api.types import CategoricalDtype
from numpy import ndarray, concatenate, stack, array, round, zeros, arange

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit

from utils.datagen import convert_df_to_array
from utils.utils import CustomProcess
from utils.constants import *

from attack_models.attack_model import PrivacyAttack

from warnings import simplefilter
simplefilter('ignore', category=FutureWarning)
simplefilter('ignore', category=DeprecationWarning)

import multiprocessing as mp

class MIAttackClassifier(PrivacyAttack):
    """"Parent class for membership inference attack on the output of a generative model using sklearn classifier"""
    def __init__(self, Distinguisher, metadata, FeatureSet=None, quids=None):

        self.Distinguisher = Distinguisher
        self.FeatureSet = FeatureSet

        self.metadata, self.categoricalAttributes, self.numericalAttributes = self._read_meta(metadata, quids)

        self.trained = False

        self.__name__ = f'{self.Distinguisher.__class__.__name__}{self.FeatureSet.__class__.__name__}'

    def train(self, synA, labels):
        """Train a membership inference attack on a labelled training set"""

        if self.FeatureSet is not None:
            synA = stack([self.FeatureSet.extract(s) for s in synA])
        else:
            synA = stack([self._df_to_array(s).flatten() for s in synA])

        if not isinstance(labels, ndarray):
            labels = array(labels)

        self.Distinguisher.fit(synA, labels)

        self.trained = True

    def attack(self, datasets, attemptLinkage=False, target=None):
        """
        Make a guess about the target's membership in the training data of the
        generative model that produced the synthetic input data

        :param datasets: list: A list of synthetic or sanitised datasets
        :return: guess: list: A guess about the target's membership for each of the synthetic input datasets
        """
        assert self.trained, 'Attack must first be trained.'

        if attemptLinkage:
            assert target is not None, 'Attacker needs target record to attempt linkage'

        guesses = []
        for df in datasets:
            if attemptLinkage:
                try:
                    k = df.groupby(self.categoricalAttributes).size()[target[self.categoricalAttributes].values]
                    if all(k == 1):
                        guess = LABEL_IN
                    else:
                        guess = self._make_guess(df)
                except:
                    guess = self._make_guess(df)
            else:
                guess = self._make_guess(df)

            guesses.append(guess)

        return guesses

    def _make_guess(self, df):
        if self.FeatureSet is not None:
            f = self.FeatureSet.extract(df).reshape(1, -1)
        else:
            f = self._df_to_array(df).reshape(1, -1)

        return round(self.Distinguisher.predict(f), 0).astype(int)[0]


    def get_confidence(self, synT, secret):
        """Calculate probability that attacker correctly predicts whether target was present in model's training data"""
        assert self.trained, 'Attack must first be trained.'
        if self.FeatureSet is not None:
            synT = stack([self.FeatureSet.extract(s) for s in synT])
        else:
            if isinstance(synT[0], DataFrame):
                synT = stack([convert_df_to_array(s, self.metadata).flatten() for s in synT])
            else:
                synT = stack([s.flatten() for s in synT])

        probs = self.Distinguisher.predict_proba(synT)

        return [p[s] for p,s in zip(probs, secret)]

    def _read_meta(self, metadata, quids):
        if quids is None:
            quids = []

        meta_dict = {}
        categoricalAttributes = []
        numericalAttributes = []

        for cdict in metadata['columns']:
            attr_name = cdict['name']
            data_type = cdict['type']

            if data_type == FLOAT or data_type == INTEGER:
                if attr_name in quids:
                    cat_bins = cdict['bins']
                    cat_labels = [f'({cat_bins[i]},{cat_bins[i+1]}]' for i in range(len(cat_bins)-1)]

                    meta_dict[attr_name] = {
                        'type': CATEGORICAL,
                        'categories': cat_labels,
                        'size': len(cat_labels)
                    }

                    categoricalAttributes.append(attr_name)

                else:
                    meta_dict[attr_name] = {
                        'type': data_type,
                        'min': cdict['min'],
                        'max': cdict['max']
                    }

                    numericalAttributes.append(attr_name)

            elif data_type == CATEGORICAL or data_type == ORDINAL:
                meta_dict[attr_name] = {
                    'type': data_type,
                    'categories': cdict['i2s'],
                    'size': len(cdict['i2s'])
                }

                categoricalAttributes.append(attr_name)

            else:
                raise ValueError(f'Unknown data type {data_type} for attribute {attr_name}')

        return meta_dict, categoricalAttributes, numericalAttributes

    def _df_to_array(self, data):
        dfAsArray = []
        for col, cdict in self.metadata.items():
            if col in list(data):
                colData = data[col].copy()
                coltype = cdict['type']

                if coltype in STRINGS:
                    if len(colData) > len(colData.dropna()):
                        colData = colData.fillna(FILLNA_VALUE_CAT)
                        if FILLNA_VALUE_CAT not in cdict['categories']:
                            col['categories'].append(FILLNA_VALUE_CAT)
                            col['size'] += 1

                    if coltype == ORDINAL:
                        cat = CategoricalDtype(categories=cdict['categories'], ordered=True)
                        colData = colData.astype(cat)
                        colArray = colData.cat.codes.values.reshape(-1, 1)

                    else:
                        colArray = self._one_hot(colData.values, cdict['categories'])

                elif coltype in NUMERICAL:
                    colArray = colData.values.reshape(-1, 1)

                else:
                    raise ValueError(f'Unknown type {coltype} for col {col}')

                dfAsArray.append(colArray)

        return concatenate(dfAsArray, axis=1)

    def _one_hot(self, col_data, categories):
        col_data_onehot = zeros((len(col_data), len(categories)))
        cidx = [categories.index(c) for c in col_data]
        col_data_onehot[arange(len(col_data)), cidx] = 1

        return col_data_onehot


class MIAttackClassifierLinearSVC(MIAttackClassifier):

    def __init__(self, metadata, FeatureSet=None):
        super().__init__(SVC(kernel='linear', probability=True), metadata, FeatureSet)


class MIAttackClassifierSVC(MIAttackClassifier):

    def __init__(self, metadata, FeatureSet=None):
        super().__init__(SVC(probability=True), metadata, FeatureSet)


class MIAttackClassifierLogReg(MIAttackClassifier):

    def __init__(self, metadata, FeatureSet=None):
        super().__init__(LogisticRegression(), metadata, FeatureSet)


class MIAttackClassifierRandomForest(MIAttackClassifier):

    def __init__(self, metadata, FeatureSet=None, quids=None):
        super().__init__(RandomForestClassifier(), metadata=metadata, FeatureSet=FeatureSet, quids=quids)


class MIAttackClassifierKNN(MIAttackClassifier):

    def __init__(self, metadata, FeatureSet=None, quids=None):
        super().__init__(KNeighborsClassifier(n_neighbors=5), metadata=metadata, FeatureSet=FeatureSet, quids=quids)


class MIAttackClassifierMLP(MIAttackClassifier):

    def __init__(self, metadata, FeatureSet=None, quids=None):
        super().__init__(MLPClassifier((200,), solver='lbfgs'), metadata=metadata, FeatureSet=FeatureSet, quids=quids)


def generate_mia_shadow_data(GenModel, target, rawA, sizeRaw, sizeSyn, numModels, numCopies):
    assert isinstance(rawA, GenModel.datatype), f"GM expects datatype {GenModel.datatype} but got {type(rawA)}"
    assert isinstance(target, type(rawA)), f"Mismatch of datatypes between target record and raw data"

    kf = ShuffleSplit(n_splits=numModels, train_size=sizeRaw)

    if GenModel.multiprocess:

        manager = mp.Manager()
        synA = manager.list()
        labelsA = manager.list()
        jobs = []
        tasks = [(rawA, train_index, GenModel, target, sizeSyn, numCopies, synA, labelsA) for train_index, _ in kf.split(rawA)]

        for task in tasks:
            p = CustomProcess(target=worker_train_shadow, args=task)
            jobs.append(p)
            p.start()

        for p in jobs:
            p.join()

    else:
        synA, labelsA = [], []
        for train_index, _ in kf.split(rawA):
            worker_train_shadow(rawA, train_index, GenModel, target, sizeSyn, numCopies, synA, labelsA)

    return synA, labelsA


def worker_train_shadow(rawA, train_index, GenModel, target, sizeSyn, numCopies, synA, labelsA):
    # Fit GM to data without target's data
    if isinstance(rawA, DataFrame):
        rawAout = rawA.iloc[train_index]
    else:
        rawAout = rawA[train_index, :]
    GenModel.fit(rawAout)

    # Generate synthetic sample for data without target
    synOut = [GenModel.generate_samples(sizeSyn) for _ in range(numCopies)]
    labelsOut = [LABEL_OUT for _ in range(numCopies)]

    # Insert targets into training data
    if isinstance(rawA, DataFrame):
        rawAin = rawAout.append(target)
    else:
        if len(target.shape) == 1:
            target = target.reshape(1, len(target))
        rawAin = concatenate([rawAout, target])

    # Fit generative model to data including target
    GenModel.fit(rawAin)

    # Generate synthetic sample for data including target
    synIn = [GenModel.generate_samples(sizeSyn) for _ in range(numCopies)]
    labelsIn = [LABEL_IN for _ in range(numCopies)]

    syn = synOut + synIn
    labels = labelsOut + labelsIn

    synA.extend(syn)
    labelsA.extend(labels)


def generate_mia_anon_data(Sanitiser, target, rawA, sizeRaw, numSamples):
    assert isinstance(rawA, Sanitiser.datatype), f"GM expects datatype {Sanitiser.datatype} but got {type(rawA)}"
    assert isinstance(target, type(rawA)), f"Mismatch of datatypes between target record and raw data"

    kf = ShuffleSplit(n_splits=numSamples, train_size=sizeRaw)

    sanA, labelsA = [], []
    for train_index, _ in kf.split(rawA):
        worker_sanitise_data(rawA, train_index, Sanitiser, target, sanA, labelsA)

    return sanA, labelsA


def worker_sanitise_data(rawA, train_index, Sanitiser, target, sanA, labelsA):
    # Fit GM to data without target's data
    if isinstance(rawA, DataFrame):
        rawAout = rawA.iloc[train_index]
    else:
        rawAout = rawA[train_index, :]
    sanOut = Sanitiser.sanitise(rawAout)
    sanA.append(sanOut)
    labelsA.append(LABEL_OUT)

    # Insert targets into training data
    if isinstance(rawA, DataFrame):
        rawAin = rawAout.append(target)
    else:
        if len(target.shape) == 1:
            target = target.reshape(1, len(target))
        rawAin = concatenate([rawAout, target])

    # Fit generative model to data including target
    sanIn = Sanitiser.sanitise(rawAin)
    sanA.append(sanIn)
    labelsA.append(LABEL_IN)



