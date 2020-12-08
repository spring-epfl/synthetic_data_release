from os import path
from pandas import DataFrame
from numpy import mean, concatenate, ndarray, ones, sqrt
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

from .privacy_attack import PrivacyAttack
from synthetic_data.utils.datagen import convert_df_to_array, convert_series_to_array
from synthetic_data.utils.logging import LOGGER


class AttributeInferenceAttack(PrivacyAttack):
    """A privacy attack that aims to reconstruct a sensitive attribute c given a partial target record T"""

    def __init__(self, RegressionModel, sensitiveAttribute, metadata, backgroundKnowledge):
        """
        Parent class for simple regression attribute inference attack

        :param RegressionModel: object: sklearn type regression model object
        :param sensitiveAttribute: string: Name of a column in a DataFrame that is considered the unknown, sensitive attribute
        :param metadata: dict: Attribute metadata describing the data domain of the synthetic target data
        :param backgroundKnowledge: DataFrame: Adversary's background knowledge dataset
        """

        self.sensitiveAttribute = sensitiveAttribute
        self.RegressionModel = RegressionModel
        self.metadata = metadata
        self.imputerCat = SimpleImputer(strategy='most_frequent')
        self.imputerNum = SimpleImputer(strategy='median')

        self.scaleFactor = None
        self.coefficients = None
        self.sigma = None

        self.priorProbabilities = self._calculate_prior_probabilities(backgroundKnowledge, self.sensitiveAttribute)
        self.trained = False

        self.__name__ = f'{self.RegressionModel.__class__.__name__}'


    def _calculate_prior_probabilities(self, backgroundKnowledge, sensitiveAttribute):
        """
        Calculate prior probability distribution over the sensitive attribuet given background knowledge

        :param backgroundKnowledge: DataFrame: Adversary's background knowledge dataset
        :param sensitiveAttribute: str: Name of a column in the DataFrame that is considered sensitive
        :return: priorProb: dict: Prior probabilities over sensitive attribute
        """

        return dict(backgroundKnowledge[sensitiveAttribute].value_counts(sort=False, dropna=False)/len(backgroundKnowledge))


    def get_prior_probability(self, sensitiveValue):
        try:
            return self.priorProbabilities[sensitiveValue]
        except:
            return 0


    def train(self, synT):
        """
        Train a MLE attack to reconstruct an unknown sensitive value from a vector of known attributes

        :param synT: DataFrame: A synthetic dataset of shape (n, k + 1)
        """

        # Split data into known and sensitive
        if isinstance(synT, DataFrame):
            assert self.sensitiveAttribute in list(synT), f'DataFrame only contains columns {list(synT)}'

            synKnown = synT.drop(self.sensitiveAttribute, axis=1)
            synSensitive = synT[self.sensitiveAttribute]

            synKnown = convert_df_to_array(synKnown, self.metadata)
            synSensitive = convert_series_to_array(synSensitive, self.metadata)

        else:
            assert isinstance(synT, ndarray), f"Unknown data type {type(synT)}"

            # If input data is array assume that self.metadata is the schema of the array
            attrList = [c['name'] for c in self.metadata['columns']]
            sensitiveIdx = attrList.index(self.sensitiveAttribute)
            synKnown = synT[:, [i for i in range(len(attrList)) if i != sensitiveIdx]]
            synSensitive = synT[:, sensitiveIdx]

        n, k = synKnown.shape

        # Centre independent variables for better regression performance
        self.scaleFactor = mean(synKnown, axis=0)
        synKnownScaled = synKnown - self.scaleFactor
        synKnownScaled = concatenate([ones((len(synKnownScaled), 1)), synKnownScaled], axis=1) # append all  ones for inclu intercept in beta vector

        # Get MLE for linear coefficients
        self.RegressionModel.fit(synKnownScaled, synSensitive)
        self.coefficients = self.RegressionModel.coef_
        self.sigma = sum((synSensitive - synKnownScaled.dot(self.coefficients))**2)/(n-k)

        LOGGER.debug('Finished training regression model')
        self.trained = True

    def attack(self, targetKnown):
        """
        Makes a guess about the target's secret attribute from the synthetic data

        :param targetKnown: ndarray or DataFrame: Partial target record with known attributes
        :return guess: float: Guess about the target's sensitive attribute value
        """

        assert self.trained, 'Attack must first be trained on some data before can predict sensitive target value'

        # Centre target record attributes
        if isinstance(targetKnown, DataFrame):
            targetKnown = convert_df_to_array(targetKnown, self.metadata)
        else:
            assert isinstance(targetKnown, ndarray), f'Unknown data type {type(targetKnown)}'

        targetKnownScaled = targetKnown - self.scaleFactor
        targetKnownScaled = concatenate([ones((len(targetKnownScaled), 1)), targetKnownScaled], axis=1)

        return targetKnownScaled.dot(self.coefficients)

    def get_likelihood(self, targetKnown, targetSensitive):
        """
        Calculate the adversary's likelihood over the target's sensitive value

        :param targetKnown: ndarray or DataFrame: Partial target record with known attributes
        :param targetSensitive: float: Target's sensitive attribute value
        :return:
        """
        assert self.trained, 'Attack must first be trained on some data before can predict sensitive target value'

        targetKnown = convert_df_to_array(targetKnown, self.metadata) # extract attribute values for known attributes
        targetKnownScaled = targetKnown - self.scaleFactor
        targetKnownScaled = concatenate([ones((len(targetKnownScaled), 1)), targetKnownScaled], axis=1)

        pdfLikelihood = norm(loc=targetKnownScaled.dot(self.coefficients), scale=sqrt(self.sigma))

        return pdfLikelihood.pdf(targetSensitive)

    def _impute_missing_values(self, df):
        """
        Impute missing values in a DataFrame

        :param df: DataFrame
        """

        catCols = list(df.select_dtypes(['object', 'category']))
        if len(catCols) > 0:
            self.imputerCat.fit(df[catCols])
            df[catCols] = self.imputerCat.transform(df[catCols])

        numCols = list(df.select_dtypes(['int', 'float']))
        if len(numCols) > 0:
            self.imputerNum.fit(df[numCols])
            df[numCols] = self.imputerNum.transform(df[numCols])

        return df


class AttributeInferenceAttackLinearRegression(AttributeInferenceAttack):
    """ An AttributeInferenceAttack based on a simple Linear Regression model """

    def __init__(self, sensitiveAttribute, metadata, backgroundKnowledge):
        super().__init__(LinearRegression(fit_intercept=False), sensitiveAttribute, metadata, backgroundKnowledge)
