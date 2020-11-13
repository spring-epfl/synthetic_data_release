from os import path
from pandas import DataFrame
from numpy import ndarray, array, linspace, all, histogram, bincount
from pandas.api.types import CategoricalDtype

from feature_sets.feature_set import FeatureSet
from utils.datagen import CATEGORICAL, CONTINUOUS, ORDINAL

import logging
from logging.config import fileConfig
dirname = path.dirname(__file__)
logconfig = path.join(dirname, '../logging_config.ini')
fileConfig(logconfig)
logger = logging.getLogger(__name__)


class HistogramFeatureSet(FeatureSet):
    def __init__(self, datatype, metadata, nbins=10):
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
            self.nfeatures += col['size']

        self.num_attr_idx = metadata['continuous_columns']
        self.num_attr_names = []
        self.histogram_bins = {}

        for cidx in self.num_attr_idx:
            col = metadata['columns'][cidx]
            self.num_attr_names.append(col['name'])
            self.histogram_bins[col['name']] = linspace(col['min'], col['max'], nbins+1)
            self.nfeatures += nbins

        logger.debug(f'Feature set will length {self.nfeatures}')

    def extract(self, data):
        assert isinstance(data, self.datatype), f'Feature extraction expects {self.datatype} as input type'
        if self.datatype is DataFrame:
            assert all([c in list(data) for c in self.cat_attr_names]), 'Missing some categorical attributes in input data'
            assert all([c in list(data) for c in self.num_attr_names]), 'Missing some numerical attributes in input data'

            features = []
            for attr in self.num_attr_names:
                col = data[attr]
                F = col.value_counts(bins=self.histogram_bins[attr]).values
                features.extend(F.tolist())
            for attr in self.cat_attr_names:
                col = data[attr]
                col = col.astype(CategoricalDtype(categories=self.category_codes[attr], ordered=True))
                F = col.value_counts().loc[self.category_codes[attr]].values
                features.extend(F.tolist())
        else:
            features = []
            for aidx, attr in zip(self.num_attr_idx, self.num_attr_names):
                col = data[:, aidx]
                F, _ = histogram(col, bins=self.histogram_bins[attr])
                features.extend(F.tolist())
            for aidx, attr in zip(self.cat_attr_idx, self.cat_attr_names):
                col = data[:, aidx].astype(int)
                F = bincount(col, minlength=len(self.category_codes[attr]))
                features.extend(F.tolist())
        assert len(features) == self.nfeatures, f'Expected number of features is {self.nfeatures} but found {len(features)}'

        return array(features)