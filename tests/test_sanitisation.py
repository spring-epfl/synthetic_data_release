"""A template file for writing a simple test for a sanitisation technique"""
from unittest import TestCase

from warnings import filterwarnings
filterwarnings('ignore')

from os import path
cwd = path.dirname(__file__)

from sanitisation_techniques.sanitiser import SanitiserNHS

from utils.datagen import load_local_data_as_df
from utils.constants import *


class TestSanitisation(TestCase):

    @classmethod
    def setUp(self) -> None:
        self.raw, self.metadata = load_local_data_as_df(path.join(cwd, 'germancredit_test'))
        self.sizeS = len(self.raw)

    def test_sanitise_nhs(self):
        print('\nTest SanitiserNHS')

        ## Test default params
        sanitiser = SanitiserNHS(self.metadata)
        san = sanitiser.sanitise(self.raw)

        # Expect no columns to be dropped or rows removed
        self.assertTupleEqual(san.shape, self.raw.shape)

        ## Test dropping columns
        sanitiser = SanitiserNHS(self.metadata, drop_cols=['Purpose'])
        san = sanitiser.sanitise(self.raw)

        # Purpose should be dropped
        self.assertTrue('Purpose' not in list(san))

        ## Test rare value threshold
        sanitiser = SanitiserNHS(self.metadata, thresh_rare=2)
        san = sanitiser.sanitise(self.raw)

        for cdict in self.metadata['columns']:
            if cdict['type'] == CATEGORICAL or cdict['type'] == ORDINAL:
                counts = san[cdict['name']].value_counts()
                self.assertTrue(len(counts[counts > 2]) == len(counts))

        ## Test converting numerical into categorical attributes
        demographics = ['Age', 'Sex', 'Job', 'Housing']
        sanitiser = SanitiserNHS(self.metadata, quids=demographics)
        san = sanitiser.sanitise(self.raw)

        self.assertListEqual([type(str) for _ in demographics], list(san[demographics].dtypes))

        ## Test k-anonymity constraint
        sanitiser = SanitiserNHS(self.metadata, quids=demographics, anonymity_set_size=7)
        san = sanitiser.sanitise(self.raw)

        counts = san.groupby(demographics).size()
        self.assertTrue(len(counts[counts >= 7]) == len(counts))


def write_to_dict(nr, results):
    results[nr] = 'a'

