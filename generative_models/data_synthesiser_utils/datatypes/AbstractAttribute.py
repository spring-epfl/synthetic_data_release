from abc import ABCMeta, abstractmethod
from bisect import bisect_right
from random import uniform

import numpy as np
from numpy.random import choice
from pandas import Series

from generative_models.data_synthesiser_utils.utils import normalize_given_distribution


class AbstractAttribute(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, data, histogram_size):
        self.name = name
        self.data = data
        self.histogram_size = histogram_size

        self.data_dropna = self.data.dropna()
        self.missing_rate = (self.data.size - self.data_dropna.size) / (self.data.size or 1)

        self.is_categorical = None
        self.is_numerical = None
        self.data_type = None
        self.min = None
        self.max = None
        self.distribution_bins = None
        self.distribution_probabilities = None
        self.domain_size = None

    def set_domain(self, domain):
        return NotImplementedError('Method needs to be overwritten.')

    @abstractmethod
    def infer_distribution(self):
        if self.is_categorical:
            histogram = self.data_dropna.value_counts()
            for value in set(self.distribution_bins) - set(histogram.index):
                histogram[value] = 0
            histogram = histogram[self.distribution_bins]
            self.distribution_probabilities = normalize_given_distribution(histogram)

        else:
            histogram, _ = np.histogram(self.data_dropna, bins=self.distribution_bins)
            self.distribution_probabilities = normalize_given_distribution(histogram)

    def encode_values_into_bin_idx(self):
        """
        Encode values into bin indices for Bayesian Network construction.
        """
        if self.is_categorical:
            value_to_bin_idx = {value: idx for idx, value in enumerate(self.distribution_bins)}
            encoded = self.data.map(lambda x: value_to_bin_idx[x], na_action='ignore')
        else:
            encoded = self.data.map(lambda x: bisect_right(self.distribution_bins[:-1], x) - 1, na_action='ignore')

        encoded.fillna(len(self.distribution_bins), inplace=True)
        return encoded.astype(int, copy=False)

    def to_json(self):
        """Encode attribution information in JSON format / Python dictionary.

        """
        return {"name": self.name,
                "data_type": self.data_type.value,
                "is_categorical": self.is_categorical,
                "min": self.min,
                "max": self.max,
                "missing_rate": self.missing_rate,
                "distribution_bins": self.distribution_bins.tolist(),
                "distribution_probabilities": self.distribution_probabilities.tolist()}

    @abstractmethod
    def generate_values_as_candidate_key(self, n):
        """When attribute should be a candidate key in output dataset.

        """
        return np.arange(n)

    def sample_binning_indices_in_independent_attribute_mode(self, n):
        """Sample an array of binning indices.

        """
        return Series(choice(len(self.distribution_probabilities), size=n, p=self.distribution_probabilities))

    @abstractmethod
    def sample_values_from_binning_indices(self, binning_indices):
        """Convert binning indices into values in domain. Used by both independent and correlated attribute mode.

        """
        return binning_indices.apply(lambda x: self.uniform_sampling_within_a_bin(x))

    def uniform_sampling_within_a_bin(self, bin_idx):
        num_bins = len(self.distribution_probabilities)
        if bin_idx == num_bins:
            return np.nan
        elif self.is_categorical:
            return self.distribution_bins[bin_idx]
        else:
            return uniform(self.distribution_bins[bin_idx], self.distribution_bins[bin_idx + 1])

