"""A set of features that a Bayesian Net model is expected to extract from the raw data"""
from os import path
from pandas import DataFrame, get_dummies
from numpy import ndarray, all, corrcoef, concatenate, nan_to_num
from pandas.api.types import CategoricalDtype

from .feature_set import FeatureSet
from .independent_histograms import HistogramFeatureSet

from synthetic_data.utils.logging import LOGGER


class CorrelationsFeatureSet(FeatureSet):
    def __init__(self, datatype, metadata):
        assert datatype in [DataFrame, ndarray], 'Unknown data type {}'.format(datatype)
        self.datatype = datatype
        self.nfeatures = 0

        self.cat_attr_idx = metadata['categorical_columns'] + metadata['ordinal_columns']
        self.cat_attr_names = []
        self.category_codes = {}

        for cidx in self.cat_attr_idx:
            col = metadata['columns'][cidx]
            self.cat_attr_names.append(col['name'])
            self.category_codes[col['name']] = col['i2s']
            self.nfeatures += col['size'] - 1

        self.num_attr_idx = metadata['continuous_columns']
        self.num_attr_names = []
        self.histogram_bins = {}

        for cidx in self.num_attr_idx:
            col = metadata['columns'][cidx]
            self.num_attr_names.append(col['name'])
            self.nfeatures += 1

        LOGGER.debug(f'Feature set will length {self.nfeatures}')

    def extract(self, data):
        assert isinstance(data, self.datatype), f'Feature extraction expects {self.datatype} as input type'
        if self.datatype is DataFrame:
            assert all([c in list(data) for c in self.cat_attr_names]), 'Missing some categorical attributes in input data'
            assert all([c in list(data) for c in self.num_attr_names]), 'Missing some numerical attributes in input data'

            encoded = data[self.num_attr_names].copy()
            for c in self.cat_attr_names:
                col = data[c]
                col = col.astype(CategoricalDtype(categories=self.category_codes[c], ordered=True))
                encoded = encoded.merge(get_dummies(col, drop_first=True), left_index=True, right_index=True)

            features = encoded.corr().fillna(0).values.flatten()

        else:
            features = nan_to_num(corrcoef(data.T).flatten())

        return features


class BayesFeatureSet(FeatureSet):
    def __init__(self, datatype, metadata, nbins=10):
        assert datatype in [DataFrame, ndarray], 'Unknown data type {}'.format(datatype)
        self.datatype = datatype

        self.histograms  = HistogramFeatureSet(datatype, metadata, nbins)
        self.correlations = CorrelationsFeatureSet(datatype, metadata)

    def extract(self, data):
        F_hist = self.histograms.extract(data)
        F_corr = self.correlations.extract(data)

        return concatenate([F_hist, F_corr])
