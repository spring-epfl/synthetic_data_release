from unittest import TestCase

from warnings import filterwarnings
filterwarnings('ignore')

from os import path
cwd = path.dirname(__file__)

from synthetic_data.feature_sets.model_agnostic import NaiveFeatureSet
from synthetic_data.feature_sets.independent_histograms import HistogramFeatureSet
from synthetic_data.feature_sets.bayes import CorrelationsFeatureSet, BayesFeatureSet
from synthetic_data.generative_models.data_synthesiser import IndependentHistogram

from synthetic_data.utils.datagen import load_local_data_as_df


class TestFeatureSets(TestCase):

    @classmethod
    def setUp(self) -> None:
        self.raw, self.metadata = load_local_data_as_df(path.join(cwd, 'germancredit_test'))
        self.sizeS = len(self.raw)

        # Test extract from synthetic data S
        GM = IndependentHistogram()
        GM.fit(self.raw)
        self.syn = GM.generate_samples(self.sizeS)

    def test_naive(self):
        # Init feature set F
        F = NaiveFeatureSet(type(self.raw))
        expected_nfeatures = 30

        # Test extract from raw data R
        feature_set = F.extract(self.raw)
        self.assertEqual(len(feature_set), expected_nfeatures)

        # Test extract from synthetic data S
        feature_set = F.extract(self.syn)
        self.assertEqual(len(feature_set), expected_nfeatures)

    def test_hist(self):
        # Init feature set F
        F = HistogramFeatureSet(type(self.raw), self.metadata)
        expected_nfeatures = 44

        # Test extract from raw data R
        feature_set = F.extract(self.raw)
        self.assertEqual(len(feature_set), expected_nfeatures)

        # Test extract from synthetic data S
        feature_set = F.extract(self.syn)
        self.assertEqual(len(feature_set), expected_nfeatures)

    def test_correlations(self):
        # Init feature set F
        F = CorrelationsFeatureSet(type(self.raw), self.metadata)
        expected_nfeatures = 100

        # Test extract from raw data R
        feature_set = F.extract(self.raw)
        self.assertEqual(len(feature_set), expected_nfeatures)

        # Test extract from synthetic data S
        feature_set = F.extract(self.syn)
        self.assertEqual(len(feature_set), expected_nfeatures)

    def test_bayes(self):
        # Init feature set F
        F = BayesFeatureSet(type(self.raw), self.metadata)
        expected_nfeatures = 144

        # Test extract from raw data R
        feature_set = F.extract(self.raw)
        self.assertEqual(len(feature_set), expected_nfeatures)

        # Test extract from synthetic data S
        feature_set = F.extract(self.syn)
        self.assertEqual(len(feature_set), expected_nfeatures)
