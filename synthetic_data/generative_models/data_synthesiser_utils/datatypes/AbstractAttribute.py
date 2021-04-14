"""
Adapted from https://github.com/DataResponsibly/DataSynthesizer

Copyright <2018> <dataresponsibly.com>

Licensed under MIT License
"""

from abc import ABCMeta, abstractmethod
from bisect import bisect_right
from random import uniform

from numpy import histogram, nan
from numpy.random import choice
from pandas import Series

from synthetic_data.generative_models.data_synthesiser_utils.utils import normalize_given_distribution


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
        """ Get marginal attribute distribution."""
        if self.is_categorical:
            frequency_counts = self.data_dropna.value_counts()
            for value in set(self.distribution_bins) - set(frequency_counts.index):
                frequency_counts[value] = 0
            frequency_counts = frequency_counts[self.distribution_bins]
            self.distribution_probabilities = normalize_given_distribution(frequency_counts)

        else:
            frequency_counts, _ = histogram(self.data_dropna, bins=self.distribution_bins)
            self.distribution_probabilities = normalize_given_distribution(frequency_counts)

    def encode_values_into_bin_idx(self):
        """ Encode values into bin indices for Bayesian Network construction."""
        if self.is_categorical:
            value_to_bin_idx = {value: idx for idx, value in enumerate(self.distribution_bins)}
            encoded = self.data.map(lambda x: value_to_bin_idx[x], na_action='ignore')
        else:
            encoded = self.data.map(lambda x: bisect_right(self.distribution_bins[:-1], x) - 1, na_action='ignore')

        encoded.fillna(len(self.distribution_bins), inplace=True)
        return encoded.astype(int, copy=False)

    def sample_binning_indices_in_independent_attribute_mode(self, n):
        """Sample an array of binning indices."""
        return Series(choice(len(self.distribution_probabilities), size=n, p=self.distribution_probabilities))

    @abstractmethod
    def sample_values_from_binning_indices(self, binning_indices):
        """Convert binning indices into values in domain. Used by both independent and correlated attribute mode."""
        return binning_indices.apply(lambda x: self.uniform_sampling_within_a_bin(x))

    def uniform_sampling_within_a_bin(self, bin_idx):
        """ Sample a value from a given histogram bin"""
        num_bins = len(self.distribution_bins)
        if bin_idx == num_bins:
            return nan
        elif self.is_categorical:
            return self.distribution_bins[bin_idx]
        else:
            return uniform(self.distribution_bins[bin_idx], self.distribution_bins[bin_idx + 1])
