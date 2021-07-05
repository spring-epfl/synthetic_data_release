"""A set of features that a Bayesian Net model is expected to extract from the raw data"""
from pandas import DataFrame, get_dummies
from pandas.api.types import CategoricalDtype
from numpy import ndarray, all, corrcoef, concatenate, nan_to_num, zeros_like, triu_indices_from
from itertools import combinations

from utils.constants import *
from utils.logging import LOGGER
from feature_sets.feature_set import FeatureSet
from feature_sets.independent_histograms import HistogramFeatureSet


class CorrelationsFeatureSet(FeatureSet):
    def __init__(self, datatype, metadata, quids=None):
        assert datatype in [DataFrame, ndarray], 'Unknown data type {}'.format(datatype)
        self.datatype = datatype
        self.nfeatures = 0

        self.cat_attributes = []
        self.num_attributes = []

        self.category_codes = {}

        if quids is None:
            quids = []

        for cdict in metadata['columns']:
            attr_name = cdict['name']
            dtype = cdict['type']

            if dtype == FLOAT or dtype == INTEGER:
                if attr_name not in quids:
                    self.num_attributes.append(attr_name)
                else:
                    self.cat_attributes.append(attr_name)
                    cat_bins = cdict['bins']
                    cat_labels = [f'({cat_bins[i]},{cat_bins[i+1]}]' for i in range(len(cat_bins)-1)]
                    self.category_codes[attr_name] = cat_labels
                    self.nfeatures += len(cat_labels)

            elif dtype == CATEGORICAL or dtype == ORDINAL:
                self.cat_attributes.append(attr_name)
                self.category_codes[attr_name] = cdict['i2s']
                self.nfeatures += len(cdict['i2s'])

        LOGGER.debug(f'Feature set will have length {self.nfeatures}')

        self.__name__ = 'Correlations'

    def extract(self, data, flatten=True):
        assert isinstance(data, self.datatype), f'Feature extraction expects {self.datatype} as input type'

        assert all([c in list(data) for c in self.cat_attributes]), 'Missing some categorical attributes in input data'
        assert all([c in list(data) for c in self.num_attributes]), 'Missing some numerical attributes in input data'

        encoded = data[self.num_attributes].copy()
        for c in self.cat_attributes:
            col = data[c]
            col = col.astype(CategoricalDtype(categories=self.category_codes[c], ordered=True))
            encoded = encoded.merge(get_dummies(col, drop_first=True, prefix=c), left_index=True, right_index=True)

        col_names = list(encoded)
        self.feature_names = list(combinations(col_names, r=2))

        corr = encoded.corr().fillna(0).values

        mask = zeros_like(corr).astype(bool)
        mask[triu_indices_from(mask, k=1)] = True

        if flatten:
            features = corr[mask].flatten()
        else:
            features = corr

        return features


class BayesFeatureSet(FeatureSet):
    def __init__(self, datatype, metadata, nbins=10, quids=None):
        assert datatype in [DataFrame, ndarray], 'Unknown data type {}'.format(datatype)
        self.datatype = datatype

        self.histograms  = HistogramFeatureSet(datatype, metadata, nbins, quids)
        self.correlations = CorrelationsFeatureSet(datatype, metadata, quids)

    def extract(self, data):
        Hist = self.histograms.extract(data)
        Corr = self.correlations.extract(data)

        return concatenate([Hist, Corr])
