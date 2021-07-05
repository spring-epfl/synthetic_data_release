from pandas import DataFrame
from numpy import ndarray, array, linspace, all
from pandas.api.types import CategoricalDtype

from feature_sets.feature_set import FeatureSet
from utils.logging import LOGGER
from utils.constants import *

from warnings import filterwarnings
filterwarnings('ignore', message=r"Parsing", category=FutureWarning)


class HistogramFeatureSet(FeatureSet):
    def __init__(self, datatype, metadata, nbins=10, quids=None):
        assert datatype in [DataFrame], 'Unknown data type {}'.format(datatype)
        self.datatype = datatype
        self.nfeatures = 0

        self.cat_attributes = []
        self.num_attributes = []

        self.histogram_bins = {}
        self.category_codes = {}

        if quids is None:
            quids = []

        for cdict in metadata['columns']:
            attr_name = cdict['name']
            dtype = cdict['type']

            if dtype == FLOAT or dtype == INTEGER:
                if attr_name not in quids:
                    self.num_attributes.append(attr_name)
                    self.histogram_bins[attr_name] = linspace(cdict['min'], cdict['max'], nbins+1)
                    self.nfeatures += nbins
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

        self.__name__ = 'Histogram'

    def extract(self, data):
        assert isinstance(data, self.datatype), f'Feature extraction expects {self.datatype} as input type'

        assert all([c in list(data) for c in self.cat_attributes]), 'Missing some categorical attributes in input data'
        assert all([c in list(data) for c in self.num_attributes]), 'Missing some numerical attributes in input data'

        features = []
        for attr in self.num_attributes:
            col = data[attr]
            F = col.value_counts(bins=self.histogram_bins[attr]).values
            features.extend(F.tolist())

        for attr in self.cat_attributes:
            col = data[attr]
            col = col.astype(CategoricalDtype(categories=self.category_codes[attr], ordered=True))
            F = col.value_counts().loc[self.category_codes[attr]].values
            features.extend(F.tolist())

        assert len(features) == self.nfeatures, f'Expected number of features is {self.nfeatures} but found {len(features)}'

        return array(features)

    def _get_names(self):
        feature_names = []
        for attr in self.num_attributes:
            bins = self.histogram_bins[attr]
            feature_names.extend([f'{attr}({int(bins[i-1])},{int(bins[i])}]' for i in range(1,len(bins))])

        for attr in self.cat_attributes:
            feature_names.extend([f'{attr}_{c}' for c in self.category_codes[attr]])

        return feature_names

