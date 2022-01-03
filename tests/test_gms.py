"""A template file for writing a simple test for a new generative model"""
from unittest import TestCase

from warnings import filterwarnings
filterwarnings('ignore')

from os import path
cwd = path.dirname(__file__)

from generative_models.data_synthesiser import IndependentHistogram, BayesianNet, PrivBayes
from generative_models.ctgan import CTGAN
from generative_models.pate_gan import PATEGAN
from generative_models.sdgym import PrivBaySDGym
from generative_models.mst_utils.mbi.dataset import Dataset, Domain
from generative_models.mst import MST

from utils.datagen import *

SEED = 42

class TestGenerativeModel(TestCase):

    @classmethod
    def setUp(self) -> None:
        self.raw, self.metadata = load_local_data_as_df(path.join(cwd, 'germancredit_test'))
        self.sizeS = len(self.raw)

    def test_independent_histogram(self):
        print('\nTest IndependentHistogram')
        ## Test default params
        gm = IndependentHistogram(self.metadata)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertListEqual(list(synthetic_data), list(self.raw))

        ## Changing nbins
        gm = IndependentHistogram(self.metadata, histogram_bins=25)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertListEqual(list(synthetic_data), list(self.raw))

    def test_bayesian_net(self):
        print('\nTest BayesianNet')
        ## Test default params
        gm = BayesianNet(self.metadata)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertListEqual(list(synthetic_data), list(self.raw))

        ## Change network degree
        gm = BayesianNet(self.metadata, degree=2)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertListEqual(list(synthetic_data), list(self.raw))

        ## Infer ranges
        gm = BayesianNet(self.metadata, infer_ranges=True)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertListEqual(list(synthetic_data), list(self.raw))

        ## Fix seed
        gm = BayesianNet(self.metadata, seed=SEED)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertListEqual(list(synthetic_data), list(self.raw))

    def test_priv_bayes(self):
        print('\nTest PrivBayes')
        ## Test default params
        gm = PrivBayes(self.metadata)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertListEqual(list(synthetic_data), list(self.raw))

        ## Change privacy param
        gm = PrivBayes(self.metadata, epsilon=1e-9)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertListEqual(list(synthetic_data), list(self.raw))

        ## Fix seed
        gm = PrivBayes(self.metadata, seed=SEED)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertListEqual(list(synthetic_data), list(self.raw))

    def test_ctgan(self):
        print('\nTest CTGAN')

        gm = CTGAN(self.metadata, batch_size=10, epochs=2)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertListEqual(list(synthetic_data), list(self.raw))


    def test_pategan(self):
        # Default params
        gm = PATEGAN(self.metadata)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertTupleEqual(synthetic_data.shape, self.raw.shape)

        # Change privacy params
        gm = PATEGAN(self.metadata, eps=10, delta=1e-1)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertTupleEqual(synthetic_data.shape, self.raw.shape)

        # Infer ranges
        gm = PATEGAN(self.metadata, infer_ranges=True)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertTupleEqual(synthetic_data.shape, self.raw.shape)

    def test_priv_bn_sdgym(self):
        print('\nTest PrivBayes SDGym')

        ## Test default params
        gm = PrivBaySDGym(self.metadata)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertListEqual(list(synthetic_data), list(self.raw))
        self.assertEqual(len(synthetic_data), self.sizeS)

        ## Repeat sampling from fitted model
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertListEqual(list(synthetic_data), list(self.raw))
        self.assertEqual(len(synthetic_data), self.sizeS)

        ## Change privacy params
        gm = PrivBaySDGym(self.metadata, epsilon=10)
        gm.fit(self.raw)
        synthetic_data = gm.generate_samples(self.sizeS)

        self.assertListEqual(list(synthetic_data), list(self.raw))
        self.assertEqual(len(synthetic_data), self.sizeS)


    def test_mst(self):
        print('\nTest MST')

        hist_bins = 25
        domain_data = {}
        cmaps = {}
        for cdict in self.metadata['columns']:
            cname = cdict['name']

            if cdict['type'] in [CATEGORICAL, ORDINAL]:
                domain_data[cname] = cdict['size']
                cmaps[cname] = {c:i for i,c in enumerate(cdict['i2s'])}

            else:
                domain_data[cname] = hist_bins

        raw_enc = self.raw.copy()

        for c, cmap in cmaps.items():
            raw_enc[c] = raw_enc[c].map(cmap)

        domain = Domain(domain_data.keys(), domain_data.values())
        dataset = Dataset(raw_enc, domain)

        epsilon = 1.0
        delta = 1e-9

        dataset_syn = MST(dataset, epsilon, delta)
        synthetic_data = dataset_syn.df

        self.assertListEqual(list(synthetic_data), list(self.raw))







