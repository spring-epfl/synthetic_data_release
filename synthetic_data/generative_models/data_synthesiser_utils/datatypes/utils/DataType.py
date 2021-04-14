"""
Adapted from https://github.com/DataResponsibly/DataSynthesizer

Copyright <2018> <dataresponsibly.com>

Licensed under MIT License
"""

from enum import Enum


class DataType(Enum):
    INTEGER = 'Integer'
    FLOAT = 'Float'
    STRING = 'String'
    DATETIME = 'DateTime'
    SOCIAL_SECURITY_NUMBER = 'SocialSecurityNumber'
