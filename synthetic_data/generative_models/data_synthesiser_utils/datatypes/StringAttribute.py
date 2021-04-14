"""
Adapted from https://github.com/DataResponsibly/DataSynthesizer

Copyright <2018> <dataresponsibly.com>

Licensed under MIT License
"""

import numpy as np

from .AbstractAttribute import AbstractAttribute
from .DataTypes import DataType

from synthetic_data.generative_models.data_synthesiser_utils.utils import normalize_given_distribution


class StringAttribute(AbstractAttribute):
    """Variable min and max are the lengths of the shortest and longest strings.

    """

    def __init__(self, name, data, histogram_size):
        super().__init__(name, data, histogram_size)
        self.is_numerical = False
        self.is_categorical = True
        self.data_type = DataType.STRING
        self.data_dropna_len = self.data_dropna.astype(str).map(len)

    def set_domain(self, domain=None):
        if domain is not None:
            lengths = [len(i) for i in domain]
            self.min = min(lengths)
            self.max = max(lengths)
            self.distribution_bins = np.array(domain)
        else:
            self.min = int(self.data_dropna_len.min())
            self.max = int(self.data_dropna_len.max())
            self.distribution_bins = self.data_dropna.unique()

        self.domain_size = len(self.distribution_bins)

    def infer_distribution(self):

        frequency_counts = self.data_dropna.value_counts()
        for value in set(self.distribution_bins) - set(frequency_counts.index):
            frequency_counts[value] = 0
        frequency_counts = frequency_counts[self.distribution_bins]

        self.distribution_probabilities = normalize_given_distribution(frequency_counts)

    def sample_values_from_binning_indices(self, binning_indices):
        return super().sample_values_from_binning_indices(binning_indices)
