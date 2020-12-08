"""
Adapted from https://github.com/DataResponsibly/DataSynthesizer

Copyright <2018> <dataresponsibly.com>

Licensed under MIT License
"""

from .AbstractAttribute import AbstractAttribute
from .utils.DataType import DataType


class IntegerAttribute(AbstractAttribute):
    def __init__(self, name, data, histogram_size, is_categorical):
        super().__init__(name, data, histogram_size, is_categorical)
        self.is_numerical = True
        self.data_type = DataType.INTEGER

    def infer_domain(self, categorical_domain=None, numerical_range=None):
        super().infer_domain(categorical_domain, numerical_range)
        self.min = int(self.min)
        self.max = int(self.max)

    def infer_distribution(self):
        super().infer_distribution()

    def generate_values_as_candidate_key(self, n):
        return super().generate_values_as_candidate_key(n)

    def sample_values_from_binning_indices(self, binning_indices):
        column = super().sample_values_from_binning_indices(binning_indices)
        column = column.round()
        column[~column.isnull()] = column[~column.isnull()].astype(int)
        return column
