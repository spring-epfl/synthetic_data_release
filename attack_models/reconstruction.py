from os import path
from pandas.api.types import CategoricalDtype
from numpy import mean, concatenate, ones, sqrt, zeros, arange
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

from attack_models.attack_model import PrivacyAttack
from utils.constants import *
from utils.logging import LOGGER


class AttributeInferenceAttack(PrivacyAttack):
    """A privacy attack that aims to reconstruct a sensitive attribute c given a partial target record T"""

    def __init__(self, PredictionModel, sensitiveAttribute, metadata, quids=None):
        """
        Parent class for simple regression attribute inference attack

        :param PredictionModel: object: sklearn-type prediction model
        :param sensitiveAttribute: string: name of a column in a DataFrame that is considered the unknown, sensitive attribute
        :param metadata: dict: schema for the data to be attacked
        :param backgroundKnowledge: pd.DataFrame: adversary's background knowledge dataset
        """

        self.PredictionModel = PredictionModel
        self.sensitiveAttribute = sensitiveAttribute

        self.metadata, self.knownAttributes, self.categoricalAttributes, self.nfeatures = self._read_meta(metadata, quids)

        self.ImputerCat = SimpleImputer(strategy='most_frequent')
        self.ImputerNum = SimpleImputer(strategy='median')

        self.trained = False

        self.__name__ = f'{self.PredictionModel.__class__.__name__}'

    def attack(self, targetAux, attemptLinkage=False, data=None):
        """Makes a guess about the target's secret attribute"""
        assert self.trained, 'Attack must first be trained on some data before can predict sensitive target value'

        if attemptLinkage:
            assert data is not None, "Need a dataset for linkage attack."
            try:
                groups = data.groupby(self.categoricalAttributes)
                targetCats = targetAux[self.categoricalAttributes].values
                groupSize = groups.size()[targetCats]
                if all(groupSize == 1):
                    guess = groups.get_group(tuple(targetCats[0]))[self.sensitiveAttribute].values[0]
                else:
                    guess = self._make_guess(targetAux)
            except:
                guess = self._make_guess(targetAux)
        else:
            guess = self._make_guess(targetAux)

        return guess

    def _make_guess(self, targetAux):
        raise NotImplementedError('Method must be overriden by a subclass')

    def _read_meta(self, metadata, quids):
        if quids is None:
            quids = []

        meta_dict = {}
        knownAttributes = []
        categoricalAttributes = []
        nfeatures = 0

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

                    nfeatures += len(cat_labels)

                    if attr_name != self.sensitiveAttribute:
                        categoricalAttributes.append(attr_name)

                else:
                    meta_dict[attr_name] = {
                        'type': data_type,
                        'min': cdict['min'],
                        'max': cdict['max']
                    }

                    nfeatures += 1

            elif data_type == CATEGORICAL or data_type == ORDINAL:
                meta_dict[attr_name] = {
                    'type': data_type,
                    'categories': cdict['i2s'],
                    'size': len(cdict['i2s'])
                }

                nfeatures += len(cdict['i2s'])

                if attr_name != self.sensitiveAttribute:
                    categoricalAttributes.append(attr_name)

            else:
                raise ValueError(f'Unknown data type {data_type} for attribute {attr_name}')

            if attr_name != self.sensitiveAttribute:
                knownAttributes.append(attr_name)

        return meta_dict, knownAttributes, categoricalAttributes, nfeatures

    def _encode_data(self, data):
        dfcopy = data.copy()
        for col, cdict in self.metadata.items():
            if col in list(dfcopy):
                col_data = dfcopy[col]
                if cdict['type'] in [CATEGORICAL, ORDINAL]:
                    if len(col_data) > len(col_data.dropna()):
                        col_data = col_data.fillna(FILLNA_VALUE_CAT)
                        if FILLNA_VALUE_CAT not in cdict['categories']:
                            col['categories'].append(FILLNA_VALUE_CAT)
                            col['size'] += 1

                    cat = CategoricalDtype(categories=cdict['categories'], ordered=True)
                    col_data = col_data.astype(cat)
                    dfcopy[col] = col_data.cat.codes

        return dfcopy.values

    def _impute_missing_values(self, df):
        dfImpute = df.copy()

        catCols = []
        numCols = []

        for attr, col in self.metadata.items():
            if attr in list(dfImpute):
                if col['type'] in [CATEGORICAL, ORDINAL]:
                    catCols.append(attr)
                elif col['type'] in NUMERICAL:
                    numCols.append(attr)

        self.ImputerCat.fit(df[catCols])
        dfImpute[catCols] = self.ImputerCat.transform(df[catCols])

        self.ImputerNum.fit(df[numCols])
        dfImpute[numCols] = self.ImputerNum.transform(df[numCols])

        return dfImpute

    def _one_hot(self, col_data, categories):
        col_data_onehot = zeros((len(col_data), len(categories)))
        cidx = [categories.index(c) for c in col_data]
        col_data_onehot[arange(len(col_data)), cidx] = 1

        return col_data_onehot


class LinRegAttack(AttributeInferenceAttack):
    """An AttributeInferenceAttack based on a simple Linear Regression model"""
    def __init__(self, sensitiveAttribute, metadata, quids=None):
        super().__init__(LinearRegression(fit_intercept=False), sensitiveAttribute, metadata, quids)

        self.scaleFactor = None
        self.coefficients = None
        self.sigma = None


    def train(self, data):
        """
        Train a MLE attack to reconstruct an unknown sensitive value from a vector of known attributes
        :param data: type(DataFrame) A dataset of shape (n, k)
        """
        features = self._encode_data(data.drop(self.sensitiveAttribute, axis=1))
        labels = data[self.sensitiveAttribute].values

        n, k = features.shape

        # Center independent variables for better regression performance
        self.scaleFactor = mean(features, axis=0)
        featuresScaled = features - self.scaleFactor
        featuresScaled = concatenate([ones((n, 1)), featuresScaled], axis=1) # append all  ones for inclu intercept in beta vector

        # Get MLE for linear coefficients
        self.PredictionModel.fit(featuresScaled, labels)
        self.coefficients = self.PredictionModel.coef_
        self.sigma = sum((labels - featuresScaled.dot(self.coefficients))**2)/(n-k)

        LOGGER.debug('Finished training regression model')
        self.trained = True

    def _make_guess(self, targetAux):
        targetFeatures = self._encode_data(targetAux)
        targetFeaturesScaled = targetFeatures - self.scaleFactor
        targetFeaturesScaled = concatenate([ones((len(targetFeaturesScaled), 1)), targetFeatures], axis=1)

        guess = targetFeaturesScaled.dot(self.coefficients)[0]

        return guess

    def get_likelihood(self, targetAux, targetSensitive, attemptLinkage=False, data=None):
        assert self.trained, 'Attack must first be trained on some data before can predict sensitive target value'

        targetFeatures = self._encode_data(targetAux)
        targetFeaturesScaled = targetFeatures - self.scaleFactor
        targetFeaturesScaled = concatenate([ones((len(targetFeaturesScaled), 1)), targetFeatures], axis=1)

        if attemptLinkage:
            assert data is not None, "Need a dataset for linkage attack."
            try:
                groups = data.groupby(self.categoricalAttributes)
                targetCats = targetAux[self.categoricalAttributes].values
                groupSize = groups.size()[targetCats]
                if all(groupSize == 1):
                    pCorrect = 1.

                else:
                    pdfLikelihood = norm(loc=targetFeaturesScaled.dot(self.coefficients), scale=sqrt(self.sigma))
                    pCorrect = pdfLikelihood.pdf(targetSensitive)[0]

            except:
                pdfLikelihood = norm(loc=targetFeaturesScaled.dot(self.coefficients), scale=sqrt(self.sigma))
                pCorrect = pdfLikelihood.pdf(targetSensitive)[0]
        else:
            pdfLikelihood = norm(loc=targetFeaturesScaled.dot(self.coefficients), scale=sqrt(self.sigma))
            pCorrect = pdfLikelihood.pdf(targetSensitive)[0]

        return pCorrect


class RandForestAttack(AttributeInferenceAttack):
    """An AttributeInferenceAttack based on a simple Linear Regression model"""
    def __init__(self, sensitiveAttribute, metadata, quids=None):
        super().__init__(RandomForestClassifier(), sensitiveAttribute, metadata, quids)

        self.labels = {l:i for i, l in enumerate(self.metadata[self.sensitiveAttribute]['categories'])}
        self.labelsInv = {i:l for l, i in self.labels.items()}

        self.scaleFactor = None

    def train(self, data):
        """
        Train a Classifier to reconstruct an unknown sensitive label from a vector of known attributes
        :param data: type(DataFrame) A dataset of shape (n, k)
        """
        features = self._encode_data(data.drop(self.sensitiveAttribute, axis=1))
        labels = data[self.sensitiveAttribute].apply(lambda x: self.labels[x]).values

        # Feature normalisation
        self.scaleFactor = mean(features, axis=0)
        featuresScaled = features - self.scaleFactor

        # Get MLE for linear coefficients
        self.PredictionModel.fit(featuresScaled, labels)

        LOGGER.debug('Finished training regression model')
        self.trained = True

    def _make_guess(self, targetAux):
        targetFeatures = self._encode_data(targetAux)
        targetFeaturesScaled = targetFeatures - self.scaleFactor

        guess = self.PredictionModel.predict(targetFeaturesScaled)

        return self.labelsInv[guess[0]]

    def get_likelihood(self, targetAux, targetSensitive, attemptLinkage=False, data=None):
        assert self.trained, 'Attack must first be trained on some data before can predict sensitive target value'

        targetFeatures = self._encode_data(targetAux)
        targetFeaturesScaled = targetFeatures - self.scaleFactor

        if attemptLinkage:
            assert data is not None, "Need a dataset for linkage attack."
            try:
                groups = data.groupby(self.categoricalAttributes)
                targetCats = targetAux[self.categoricalAttributes].values
                groupSize = groups.size()[targetCats]
                if all(groupSize == 1):
                    pCorrect = 1.

                else:
                    probs = self.PredictionModel.predict_proba(targetFeaturesScaled).flatten()
                    pCorrect = probs[self.labels[targetSensitive]]

            except:
                probs = self.PredictionModel.predict_proba(targetFeaturesScaled).flatten()
                pCorrect = probs[self.labels[targetSensitive]]
        else:
            probs = self.PredictionModel.predict_proba(targetFeaturesScaled).flatten()
            pCorrect = probs[self.labels[targetSensitive]]

        return pCorrect