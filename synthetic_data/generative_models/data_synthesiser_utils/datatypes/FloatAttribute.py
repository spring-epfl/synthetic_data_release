"""
Adapted from https://github.com/DataResponsibly/DataSynthesizer

Copyright <2018> <dataresponsibly.com>

Licensed under MIT License
"""

from numpy import arange, linspace, histogram

from .AbstractAttribute import AbstractAttribute
from .DataTypes import DataType

from synthetic_data.generative_models.data_synthesiser_utils.utils import normalize_given_distribution

class FloatAttribute(AbstractAttribute):
    def __init__(self, name, data, histogram_size):
        super().__init__(name, data, histogram_size)
        self.is_categorical = False
        self.is_numerical = True
        self.data_type = DataType.FLOAT
        self.data = self.data_type.astype(float)
        self.data_dropna = self.data_dropna.astype(float)

    def set_domain(self, domain=None):
        if domain is not None:
            self.min, self.max = domain
        else:
            self.min = float(self.data_dropna.min())
            self.max = float(self.data_dropna.max())

        self.distribution_bins = linspace(self.min, self.max, self.histogram_size+1)
        self.domain_size = self.histogram_size

    def infer_distribution(self):
        frequency_counts, _ = histogram(self.data_dropna, bins=self.distribution_bins)
        self.distribution_probabilities = normalize_given_distribution(frequency_counts)

    def sample_values_from_binning_indices(self, binning_indices):
        return super().sample_values_from_binning_indices(binning_indices)
