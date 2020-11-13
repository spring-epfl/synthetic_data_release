"""A template file for writing a simple test for a new attack model"""
from unittest import TestCase, skip

from warnings import filterwarnings
filterwarnings('ignore')

from os import path
cwd = path.dirname(__file__)

from numpy import all
from sklearn.model_selection import train_test_split

from privacy_attacks.membership_inference import *
from privacy_attacks.attribute_inference import *

from generative_models.data_synthesiser import IndependentHistogram
from feature_sets.model_agnostic import NaiveFeatureSet
from utils.datagen import load_local_data_as_df

NUM_SHADOWS = 2
NUM_SYN_COPIES = 10
PRIOR = {LABEL_IN: 0.5, LABEL_OUT: 0.5}
SENSITIVE = 'Duration'

class TestAttacks(TestCase):

    @classmethod
    def setUp(self):
        raw, self.metadata = load_local_data_as_df(path.join(cwd, 'germancredit_test'))
        self.rawA, self.rawTest = train_test_split(raw, test_size=.25)
        self.sizeS = self.sizeR = len(self.rawTest)
        self.target = self.rawTest.iloc[1, :].to_frame().T

        self.GM = IndependentHistogram()

        self.synA, self.labelsSynA = generate_mia_shadow_data_shufflesplit(self.GM, self.target, self.rawA,
                                                                           self.sizeR, self.sizeS,
                                                                           NUM_SHADOWS, NUM_SYN_COPIES)

        self.FeatureSet =  NaiveFeatureSet(DataFrame)

    def test_mia_svc(self):
        print('\n--Test MIA with non-linear SVC--')

        # Test without feature extraction
        Attack = MIAttackClassifierSVC(self.metadata, PRIOR)
        Attack.train(self.synA, self.labelsSynA)

        self.GM.fit(self.rawTest)
        synTest = [self.GM.generate_samples(self.sizeS) for _ in range(NUM_SYN_COPIES)]
        guesses = Attack.attack(synTest)
        probSuccess = Attack.get_probability_of_success(synTest, [LABEL_IN for _ in range(NUM_SYN_COPIES)])

        self.assertEqual(len(guesses), len(synTest))
        self.assertTrue(all([p in [0, 1] for p in guesses]))
        self.assertTrue(all([0 <= p <= 1. for p in probSuccess]))

        print(f'MIA accuracy without feature extraction: {sum([g == LABEL_IN for g in guesses])/NUM_SYN_COPIES}')

        # Test with feature extraction
        Attack = MIAttackClassifierSVC(self.metadata, PRIOR, self.FeatureSet)
        Attack.train(self.synA, self.labelsSynA)

        self.GM.fit(self.rawTest)
        synTest = [self.GM.generate_samples(self.sizeS) for _ in range(NUM_SYN_COPIES)]
        guesses = Attack.attack(synTest)
        probSuccess = Attack.get_probability_of_success(synTest, [LABEL_IN for _ in range(NUM_SYN_COPIES)])

        self.assertEqual(len(guesses), len(synTest))
        self.assertTrue(all([p in [0, 1] for p in guesses]))
        self.assertTrue(all([0 <= p <= 1. for p in probSuccess]))

        print(f'MIA accuracy with naive feature set: {sum([g == LABEL_IN for g in guesses])/NUM_SYN_COPIES}')

    @skip('Does not converge with small test dataset')
    def test_mia_linear_svc(self):
        print('\n--Test MIA with linear SVC--')
        # Test without feature extraction
        Attack = MIAttackClassifierLinearSVC(self.metadata, PRIOR)
        Attack.train(self.synA, self.labelsSynA)

        self.GM.fit(self.rawTest)
        synTest = [self.GM.generate_samples(self.sizeS) for _ in range(NUM_SYN_COPIES)]
        guesses = Attack.attack(synTest)
        probSuccess = Attack.get_probability_of_success(synTest, [LABEL_IN for _ in range(NUM_SYN_COPIES)])

        self.assertEqual(len(guesses), len(synTest))
        self.assertTrue(all([p in [0, 1] for p in guesses]))
        self.assertTrue(all([0 <= p <= 1. for p in probSuccess]))

        print(f'MIA accuracy without feature extraction: {sum([g == LABEL_IN for g in guesses])/NUM_SYN_COPIES}')

        # Test with feature extraction
        Attack = MIAttackClassifierLinearSVC(self.metadata, PRIOR, self.FeatureSet)
        Attack.train(self.synA, self.labelsSynA)

        self.GM.fit(self.rawTest)
        synTest = [self.GM.generate_samples(self.sizeS) for _ in range(NUM_SYN_COPIES)]
        guesses = Attack.attack(synTest)
        probSuccess = Attack.get_probability_of_success(synTest, [LABEL_IN for _ in range(NUM_SYN_COPIES)])

        self.assertEqual(len(guesses), len(synTest))
        self.assertTrue(all([p in [0, 1] for p in guesses]))
        self.assertTrue(all([0 <= p <= 1. for p in probSuccess]))

        print(f'MIA accuracy with naive features: {sum([g == LABEL_IN for g in guesses])/NUM_SYN_COPIES}')

    def test_mia_randforest(self):
        print('\n--Test MIA with RandForest--')
        # Test without feature extraction
        Attack = MIAttackClassifierRandomForest(self.metadata, PRIOR)
        Attack.train(self.synA, self.labelsSynA)

        self.GM.fit(self.rawTest)
        synTest = [self.GM.generate_samples(self.sizeS) for _ in range(NUM_SYN_COPIES)]
        guesses = Attack.attack(synTest)
        probSuccess = Attack.get_probability_of_success(synTest, [LABEL_IN for _ in range(NUM_SYN_COPIES)])

        self.assertEqual(len(guesses), len(synTest))
        self.assertTrue(all([p in [0, 1] for p in guesses]))
        self.assertTrue(all([0 <= p <= 1. for p in probSuccess]))

        print(f'MIA accuracy without feature extraction: {sum([g == LABEL_IN for g in guesses])/NUM_SYN_COPIES}')

        # Test with feature extraction
        Attack = MIAttackClassifierRandomForest(self.metadata, PRIOR, self.FeatureSet)
        Attack.train(self.synA, self.labelsSynA)

        self.GM.fit(self.rawTest)
        synTest = [self.GM.generate_samples(self.sizeS) for _ in range(NUM_SYN_COPIES)]
        guesses = Attack.attack(synTest)
        probSuccess = Attack.get_probability_of_success(synTest, [LABEL_IN for _ in range(NUM_SYN_COPIES)])

        self.assertEqual(len(guesses), len(synTest))
        self.assertTrue(all([p in [0, 1] for p in guesses]))
        self.assertTrue(all([0 <= p <= 1. for p in probSuccess]))

        print(f'MIA accuracy with naive feature set: {sum([g == LABEL_IN for g in guesses])/NUM_SYN_COPIES}')

    def test_logreg(self):
        print('\n--Test MIA with LogReg--')
        # Test without feature extraction
        Attack = MIAttackClassifierLogReg(self.metadata, PRIOR)
        Attack.train(self.synA, self.labelsSynA)

        self.GM.fit(self.rawTest)
        synTest = [self.GM.generate_samples(self.sizeS) for _ in range(NUM_SYN_COPIES)]
        guesses = Attack.attack(synTest)
        probSuccess = Attack.get_probability_of_success(synTest, [LABEL_IN for _ in range(NUM_SYN_COPIES)])

        self.assertEqual(len(guesses), len(synTest))
        self.assertTrue(all([p in [0, 1] for p in guesses]))
        self.assertTrue(all([0 <= p <= 1. for p in probSuccess]))

        print(f'MIA accuracy without feature extraction: {sum([g == LABEL_IN for g in guesses])/NUM_SYN_COPIES}')

        # Test with feature extraction
        Attack = MIAttackClassifierLogReg(self.metadata, PRIOR, self.FeatureSet)
        Attack.train(self.synA, self.labelsSynA)

        self.GM.fit(self.rawTest)
        synTest = [self.GM.generate_samples(self.sizeS) for _ in range(NUM_SYN_COPIES)]
        guesses = Attack.attack(synTest)
        probSuccess = Attack.get_probability_of_success(synTest, [LABEL_IN for _ in range(NUM_SYN_COPIES)])

        self.assertEqual(len(guesses), len(synTest))
        self.assertTrue(all([p in [0, 1] for p in guesses]))
        self.assertTrue(all([0 <= p <= 1. for p in probSuccess]))

        print(f'MIA accuracy with naive feature sets: {sum([g == LABEL_IN for g in guesses])/NUM_SYN_COPIES}')

    def test_attr_linreg(self):
        print('\n--Test AttributeInference with LinReg--')

        targetKnown = self.target.drop(SENSITIVE, axis=1)
        secret = self.target[SENSITIVE].values
        Attack = AttributeInferenceAttackLinearRegression(SENSITIVE, self.metadata, self.rawA)

        Attack.train(self.rawTest)
        guess = Attack.attack(targetKnown)

        print(f'True value: {secret}, Guess R: {guess}')

        self.GM.fit(self.rawTest)
        synTest = self.GM.generate_samples(self.sizeS)

        Attack.train(synTest)
        guess = Attack.attack(targetKnown)

        print(f'True value: {secret}, Guess S: {guess}')









