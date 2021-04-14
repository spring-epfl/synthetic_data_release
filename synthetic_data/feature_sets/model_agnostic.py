"""A simple feature extraction layer for data with a mix of categorical and numerical attributes"""
from os import path
from pandas import DataFrame
from numpy import ndarray, nanmean, nanmedian, nanvar, array, concatenate
from pandas.api.types import is_numeric_dtype, CategoricalDtype

from .feature_set import FeatureSet
from .independent_histograms import HistogramFeatureSet
from .bayes import CorrelationsFeatureSet


class NaiveFeatureSet(FeatureSet):
    def __init__(self, datatype):
        self.datatype = datatype
        self.attributes = None
        self.category_codes = {}
        assert self.datatype in [DataFrame, ndarray], 'Unknown data type {}'.format(datatype)

    def extract(self, data):
        if self.datatype is DataFrame:
            assert isinstance(data, DataFrame), 'Feature extraction expects DataFrame as input'
            if self.attributes is not None:
                if bool(set(list(data)).difference(set(self.attributes))):
                    raise ValueError('Data to filter does not match expected schema')
            else:
                self.attributes = list(data)
            features = DataFrame(columns=self.attributes)
            for c in self.attributes:
                col = data[c]
                if is_numeric_dtype(col):
                    features[c] = [col.mean(), col.median(), col.var()]
                else:
                    if c in self.category_codes.keys():
                        new_cats = set(col.astype('category').cat.categories).difference(set(self.category_codes[c]))
                        self.category_codes[c] += list(new_cats)
                        col = col.astype(CategoricalDtype(categories=self.category_codes[c]))
                    else:
                        col = col.astype('category')
                        self.category_codes[c] = list(col.cat.categories)
                    counts = list(col.cat.codes.value_counts().index)
                    features[c] = [counts[0], counts[-1], len(counts)]
            features = features.values

        elif self.datatype is ndarray:
            assert isinstance(data, ndarray), 'Feature extraction expects ndarray as input'
            features = array([nanmean(data), nanmedian(data), nanvar(data)])
        else:
            raise ValueError(f'Unknown data type {type(data)}')

        return features.flatten()


class EnsembleFeatureSet(FeatureSet):
    """An ensemble of features that is not model specific"""
    def __init__(self, datatype, metadata, nbins=10):
        assert datatype in [DataFrame, ndarray], 'Unknown data type {}'.format(datatype)
        self.datatype = datatype

        self.naive = NaiveFeatureSet(datatype)
        self.histograms  = HistogramFeatureSet(datatype, metadata, nbins)
        self.correlations = CorrelationsFeatureSet(datatype, metadata)

    def extract(self, data):
        F_naive = self.naive.extract(data)
        F_hist = self.histograms.extract(data)
        F_corr = self.correlations.extract(data)

        return concatenate([F_naive, F_hist, F_corr])







