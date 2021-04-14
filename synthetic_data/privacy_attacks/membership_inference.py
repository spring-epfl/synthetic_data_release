""" Membership inference attack on synthetic data that implements the risk of linkability. """
from copy import deepcopy
from pandas import DataFrame
from numpy import ndarray, concatenate, stack, array, round
from multiprocessing.pool import Pool

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit

from synthetic_data.utils.constants import *
from synthetic_data.utils.datagen import convert_df_to_array
from synthetic_data.utils.logging import LOGGER
from .privacy_attack import PrivacyAttack


class MIAttackClassifier(PrivacyAttack):
    """" Parent class for membership inference attacks based on shadow modelling
    using a sklearn classifier as attack model """

    def __init__(self, AttackClassifier, metadata, priorProbabilities, FeatureSet=None):
        """

        :param AttackClassifier: Classifier: An object that implements a binary classifier
        :param metadata: dict: Attribute metadata describing the data domain of the synthetic target data
        :param priorProbabilities: dict: Prior probabilities over the target's membership
        :param FeatureSet: FeatureSet: An object that implements a feacture extraction strategy for converting a dataset into a feature vector
        """

        self.AttackClassifier = AttackClassifier
        self.FeatureSet = FeatureSet
        self.ImputerCat = SimpleImputer(strategy='most_frequent')
        self.ImputerNum = SimpleImputer(strategy='median')
        self.metadata = metadata

        self.priorProbabilities = priorProbabilities

        self.trained = False

        self.__name__ = f'{self.AttackClassifier.__class__.__name__}{self.FeatureSet.__class__.__name__}'

    def _get_prior_probability(self, secret):
        """ Get prior probability of the adversary guessing the target's secret

        :param secret: int: Target's true secret. Either LABEL_IN=1 or LABEL_OUT=0
        """
        try:
            return self.priorProbabilities[secret]
        except:
            return 0

    def train(self, synA, labels):
        """ Train a membership inference attack on a labelled training set

         :param synA: list of ndarrays: A list of synthetic datasets
         :param labels: list: A list of labels that indicate whether target was in the training data (LABEL_IN=1) or not (LABEL_OUT=0)
         """

        if self.FeatureSet is not None:
            synA = stack([self.FeatureSet.extract(s) for s in synA])
        else:
            if isinstance(synA[0], DataFrame):
                synA = [self._impute_missing_values(s) for s in synA]
                synA = stack([convert_df_to_array(s, self.metadata).flatten() for s in synA])
            else:
                synA = stack([s.flatten() for s in synA])
        if not isinstance(labels, ndarray):
            labels = array(labels)

        self.AttackClassifier.fit(synA, labels)

        LOGGER.debug('Finished training MIA distinguisher')
        self.trained = True

        del synA, labels

    def attack(self, synT):
        """ Makes a guess about target's presence in the training set of the model that produced the synthetic input data

        :param synT: ndarray or DataFrame: A synthetic dataset
        """

        assert self.trained, 'Attack must first be trained before can predict membership'

        if self.FeatureSet is not None:
            synT = stack([self.FeatureSet.extract(s) for s in synT])
        else:
            if isinstance(synT[0], DataFrame):
                synT = stack([convert_df_to_array(s, self.metadata).flatten() for s in synT])
            else:
                synT = stack([s.flatten() for s in synT])

        return round(self.AttackClassifier.predict(synT), 0).astype(int).tolist()

    def get_probability_of_success(self, synT, secret):
        """Calculate probability that attacker correctly predicts whether target was present in model's training data

        :param synT: ndarray or DataFrame: A synthetic dataset
        :param secret: int: Target's true secret. Either LABEL_IN=1 or LABEL_OUT=0
        """

        assert self.trained, 'Attack must first be trained on some random data before can predict membership of target data'

        if self.FeatureSet is not None:
            synT = stack([self.FeatureSet.extract(s) for s in synT])
        else:
            if isinstance(synT[0], DataFrame):
                synT = stack([convert_df_to_array(s, self.metadata).flatten() for s in synT])
            else:
                synT = stack([s.flatten() for s in synT])

        probs = self.AttackClassifier.predict_proba(synT)

        return [p[s] for p,s in zip(probs, secret)]

    def _impute_missing_values(self, df):
        """ Impute missing values in a DataFrame

        :param df: DataFrame
        """

        cat_cols = list(df.select_dtypes(['object', 'category']))
        if len(cat_cols) > 0:
            self.ImputerCat.fit(df[cat_cols])
            df[cat_cols] = self.ImputerCat.transform(df[cat_cols])

        num_cols = list(df.select_dtypes(['int', 'float']))
        if len(num_cols) > 0:
            self.ImputerNum.fit(df[num_cols])
            df[num_cols] = self.ImputerNum.transform(df[num_cols])

        return df


class MIAttackClassifierLinearSVC(MIAttackClassifier):
    """ Membership inference attack based on shadow modelling using a linear SVClassifier """
    def __init__(self, metadata, priorProbabilities, FeatureSet=None):
        super().__init__(SVC(kernel='linear', probability=True), metadata, priorProbabilities, FeatureSet)


class MIAttackClassifierSVC(MIAttackClassifier):
    """ Membership inference attack based on shadow modelling using a non-linear SVClassifier"""
    def __init__(self, metadata, priorProbabilities, FeatureSet=None):
        super().__init__(SVC(probability=True), metadata, priorProbabilities, FeatureSet)


class MIAttackClassifierLogReg(MIAttackClassifier):
    """ Membership inference attack based on shadow modelling using a LogisticRegression Classifier"""
    def __init__(self, metadata, priorProbabilities, FeatureSet=None):
        super().__init__(LogisticRegression(), metadata, priorProbabilities, FeatureSet)


class MIAttackClassifierRandomForest(MIAttackClassifier):
    """ Membership inference attack based on shadow modelling using a RandomForestClassifier"""
    def __init__(self, metadata, priorProbabilities, FeatureSet=None):
        super().__init__(RandomForestClassifier(), metadata, priorProbabilities, FeatureSet)


class MIAttackClassifierKNN(MIAttackClassifier):
    """ Membership inference attack based on shadow modelling using a KNeighborsClassifier """
    def __init__(self, metadata, priorProbabilities, FeatureSet=None):
        super().__init__(KNeighborsClassifier(n_neighbors=5), metadata, priorProbabilities, FeatureSet)


class MIAttackClassifierMLP(MIAttackClassifier):
    """ Membership inference attack based on shadow modelling using a multi-layer perceptron as classifier"""
    def __init__(self, metadata, priorProbabilities, FeatureSet=None):
        super().__init__(MLPClassifier((200,), solver='lbfgs'), metadata, priorProbabilities, FeatureSet)


def generate_mia_shadow_data_shufflesplit(GenModel, target, rawA, sizeRaw, sizeSyn, numModels, numCopies, multiprocess=False):
    """ Procedure to train a set of shadow models on multiple training sets sampled from a reference dataset.

    :param GenModel: GenerativeModel: An object that implements a generative model training procedure
    :param target: ndarray or DataFrame: The target record
    :param rawA: ndarray or DataFrame: Attacker's reference dataset of size n_A
    :param sizeRaw: int: Size of the target training set
    :param sizeSyn: int: Size of the synthetic dataset the adversary will be given access to
    :param numModels: int: Number of shadow models to train
    :param numCopies: int: Number of synthetic training datasets sampled from each shadow model

    :returns
        :return synA: list of ndarrays or DataFrames: List of synthetic datasets
        :return labels: list: List of labels indicating whether target was in or out
    """
    assert isinstance(rawA, GenModel.datatype), f"GM expectes datatype {GenModel.datatype} but got {type(rawA)}"
    assert isinstance(target, type(rawA)), f"Mismatch of datatypes between target record and raw data"

    kf = ShuffleSplit(n_splits=numModels, train_size=sizeRaw)
    synA, labels = [], []

    LOGGER.debug(f'Start training {numModels} shadow models of class {GenModel.__name__}')
    tasks = [(rawA, train_index, deepcopy(GenModel), target, sizeSyn, numCopies) for train_index, _ in kf.split(rawA)]

    if multiprocess:
        with Pool(processes=PROCESSES) as pool:
            resultsList = pool.map(worker_train_shadow, tasks)
    else:
        resultsList = []
        for task in tasks:
            res = worker_train_shadow(task)
            resultsList.append(res)

    for res in resultsList:
        s, l = res
        synA.extend(s)
        labels.extend(l)

    return synA, labels


def worker_train_shadow(params):
    rawA, train_index, GenModel, target, sizeSyn, numCopies = params

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

    return syn, labels