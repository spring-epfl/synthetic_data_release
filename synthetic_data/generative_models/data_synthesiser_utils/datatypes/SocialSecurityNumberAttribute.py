"""
Adapted from https://github.com/DataResponsibly/DataSynthesizer

Copyright <2018> <dataresponsibly.com>

Licensed under MIT License
"""

import numpy as np
from pandas import Series

from .AbstractAttribute import AbstractAttribute
from .utils.DataType import DataType


def pre_process(column: Series):
    if column.size == 0:
        return column
    elif type(column.iloc[0]) is int:
        return column
    elif type(column.iloc[0]) is str:
        return column.map(lambda x: int(x.replace('-', '')))
    else:
        raise Exception('Invalid SocialSecurityNumber.')


def is_ssn(value):
    """Test whether a number is between 0 and 1e9.

    Note this function does not take into consideration some special numbers that are never allocated.
    https://en.wikipedia.org/wiki/Social_Security_number
    """
    if type(value) is int:
        return 0 < value < 1e9
    elif type(value) is str:
        value = value.replace('-', '')
        if value.isdigit():
            return 0 < int(value) < 1e9
    return False


class SocialSecurityNumberAttribute(AbstractAttribute):
    """SocialSecurityNumber of format AAA-GG-SSSS.

    """

    def __init__(self, name, data, histogram_size, is_categorical):
        super().__init__(name, pre_process(data), histogram_size, is_categorical)
        self.is_numerical = True
        self.data_type = DataType.SOCIAL_SECURITY_NUMBER

    def infer_domain(self, categorical_domain=None, numerical_range=None):
        super().infer_domain(categorical_domain, numerical_range)
        self.min = int(self.min)
        self.max = int(self.max)

    def infer_distribution(self):
        super().infer_distribution()

    def generate_values_as_candidate_key(self, n):
        if n < 1e9:
            values = np.linspace(0, 1e9 - 1, num=n, dtype=int)
            values = np.random.permutation(values)
            values = [str(i).zfill(9) for i in values]
            return ['{}-{}-{}'.format(i[:3], i[3:5], i[5:]) for i in values]
        else:
            raise Exception('The candidate key "{}" cannot generate more than 1e9 distinct values.', self.name)

    def sample_values_from_binning_indices(self, binning_indices):
        return super().sample_binning_indices_in_independent_attribute_mode(binning_indices)
