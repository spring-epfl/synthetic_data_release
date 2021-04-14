"""
Adapted from https://github.com/DataResponsibly/DataSynthesizer

Copyright <2018> <dataresponsibly.com>

Licensed under MIT License
"""

from pandas import Series

from synthetic_data.generative_models.data_synthesiser_utils.datatypes.DateTimeAttribute import DateTimeAttribute
from synthetic_data.generative_models.data_synthesiser_utils.datatypes.FloatAttribute import FloatAttribute
from synthetic_data.generative_models.data_synthesiser_utils.datatypes.IntegerAttribute import IntegerAttribute
from synthetic_data.generative_models.data_synthesiser_utils.datatypes.SocialSecurityNumberAttribute import SocialSecurityNumberAttribute
from synthetic_data.generative_models.data_synthesiser_utils.datatypes.StringAttribute import StringAttribute
from synthetic_data.generative_models.data_synthesiser_utils.datatypes.constants import *
from .DataType import DataType

def parse_json(attribute_in_json):
    name = attribute_in_json['name']
    data_type = DataType(attribute_in_json['data_type'])
    is_categorical = attribute_in_json['is_categorical']
    histogram_size = len(attribute_in_json['distribution_bins'])
    if data_type is DataType.INTEGER:
        attribute = IntegerAttribute(name, Series(), histogram_size, is_categorical)
    elif data_type is DataType.FLOAT:
        attribute = FloatAttribute(name, Series(), histogram_size, is_categorical)
    elif data_type is DataType.DATETIME:
        attribute = DateTimeAttribute(name, Series(), histogram_size, is_categorical)
    elif data_type is DataType.STRING:
        attribute = StringAttribute(name, Series(), histogram_size, is_categorical)
    elif data_type is DataType.SOCIAL_SECURITY_NUMBER:
        attribute = SocialSecurityNumberAttribute(name, Series(), histogram_size, is_categorical)
    else:
        raise Exception('Data type {} is unknown.'.format(data_type.value))

    attribute.missing_rate = attribute_in_json['missing_rate']
    attribute.min = attribute_in_json['min']
    attribute.max = attribute_in_json['max']
    attribute.distribution_bins = attribute_in_json['distribution_bins']
    attribute.distribution_probabilities = attribute_in_json['distribution_probabilities']

    return attribute
