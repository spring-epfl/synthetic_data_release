"""A template file for writing a simple test for a new generative model"""
from unittest import skip, TestCase

from warnings import filterwarnings
filterwarnings('ignore')

from os import path
cwd = path.dirname(__file__)

from synthetic_data.generative_models.data_synthesiser import IndependentHistogram, BayesianNet, PrivBayes
from synthetic_data.generative_models.ctgan import CTGAN
from synthetic_data.generative_models.pate_gan import PateGan


from synthetic_data.utils.datagen import *


class TestGenerativeModel(TestCase):

    @classmethod
    def setUp(self) -> None:
        self.raw, self.metadata = load_local_data_as_df(path.join(cwd, 'germancredit_test'))
        self.sizeS = len(self.raw)

    def test_independent_histogram(self):
        print('\nTest IndHist')

        gm = IndependentHistogram(self.metadata)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertTupleEqual(synthetic_data.shape, self.raw.shape)

    def test_bayesian_net(self):
        print('\nTest BayNet')

        # Default params
        gm = BayesianNet(self.metadata)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertTupleEqual(synthetic_data.shape, self.raw.shape)
        self.assertTrue(all([c in list(synthetic_data) for c in list(self.raw)]))
        print('Passed with default params.')

        # Degree > 1
        gm = BayesianNet(self.metadata, k=2)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertTupleEqual(synthetic_data.shape, self.raw.shape)
        self.assertTrue(all([c in list(synthetic_data) for c in list(self.raw)]))
        print('Passed with degree > 1.')

        # Multiprocess
        gm = BayesianNet(self.metadata, multiprocess=True)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertTupleEqual(synthetic_data.shape, self.raw.shape)
        self.assertTrue(all([c in list(synthetic_data) for c in list(self.raw)]))
        print('Passed with multi-processing.')

    def test_priv_bayes(self):
        print('\nTest PrivBay')

        # Default
        gm = PrivBayes(self.metadata)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertTupleEqual(synthetic_data.shape, self.raw.shape)
        self.assertTrue(all([c in list(synthetic_data) for c in list(self.raw)]))
        print('Passed with default params.')

        # Degree > 1, decrease privacy
        gm = PrivBayes(self.metadata, k=2, epsilon=10)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertTupleEqual(synthetic_data.shape, self.raw.shape)
        self.assertTrue(all([c in list(synthetic_data) for c in list(self.raw)]))
        print('Passed with degree > 1.')

        # Multi-process
        gm = PrivBayes(self.metadata, multiprocess=True)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertTupleEqual(synthetic_data.shape, self.raw.shape)
        self.assertTrue(all([c in list(synthetic_data) for c in list(self.raw)]))
        print('Passed with multiprocessing.')

    def test_ctgan(self):
        print('\nTest CTGAN')

        gm = CTGAN(self.metadata, batch_size=10, epochs=2)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertTupleEqual(synthetic_data.shape, self.raw.shape)
        self.assertTrue(all([c in list(synthetic_data) for c in list(self.raw)]))

    @skip(reason="")
    def test_pategan(self):
        print('\nTest PateGan')

        gm = PateGan(self.metadata, eps=0, delta=0)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertTupleEqual(synthetic_data.shape, self.raw.shape)
        self.assertTrue(all([c in list(synthetic_data) for c in list(self.raw)]))








