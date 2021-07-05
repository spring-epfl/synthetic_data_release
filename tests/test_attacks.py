"""A template file for writing a simple test for a new attack model"""
from unittest import TestCase
from pandas import DataFrame

from warnings import filterwarnings
filterwarnings('ignore')

from os import path
cwd = path.dirname(__file__)

from attack_models.mia_classifier import (MIAttackClassifierLinearSVC,
                                          MIAttackClassifierLogReg,
                                          MIAttackClassifierRandomForest,
                                          generate_mia_shadow_data,
                                          generate_mia_anon_data)

from generative_models.data_synthesiser import IndependentHistogram
from sanitisation_techniques.sanitiser import SanitiserNHS
from feature_sets.independent_histograms import HistogramFeatureSet
from utils.datagen import load_local_data_as_df

class TestAttacks(TestCase):
    @classmethod
    def setUp(self) -> None:
        self.raw, self.metadata = load_local_data_as_df(path.join(cwd, 'germancredit_test'))
        self.sizeS = int(len(self.raw)/2)
        self.GenModel = IndependentHistogram(self.metadata)
        self.San = SanitiserNHS(self.metadata)
        self.FeatureSet = HistogramFeatureSet(DataFrame, metadata=self.metadata)

        self.target = self.raw.sample()
        self.shadowDataSyn = generate_mia_shadow_data(self.GenModel, self.target, self.raw, self.sizeS, self.sizeS, numModels=2, numCopies=2)
        self.shadowDataSan = generate_mia_anon_data(self.San, self.target, self.raw, self.sizeS, numSamples=2)

        self.GenModel.fit(self.raw)
        self.synthetic = [self.GenModel.generate_samples(self.sizeS) for _ in range(10)]
        self.sanitised = [self.San.sanitise(self.raw) for _ in range(10)]

    def test_mia_randforest(self):
        print('\nTest MIA RandForest')
        ## Default without feature extraction
        Attack = MIAttackClassifierRandomForest(metadata=self.metadata)
        Attack.train(*self.shadowDataSyn)

        guesses = Attack.attack(self.synthetic)
        self.assertEqual(len(guesses), len(self.synthetic))

        ## With FeatureSet
        Attack = MIAttackClassifierRandomForest(metadata=self.metadata, FeatureSet=self.FeatureSet)
        Attack.train(*self.shadowDataSyn)

        guesses = Attack.attack(self.synthetic)
        self.assertEqual(len(guesses), len(self.synthetic))

        ## Test linkage
        Attack.train(*self.shadowDataSan)
        guesses = Attack.attack(self.sanitised, attemptLinkage=True, target=self.target)
        self.assertEqual(len(guesses), len(self.sanitised))


    def test_mia_logreg(self):
        print('\nTest MIA LogReg')
        Attack = MIAttackClassifierLogReg(metadata=self.metadata, FeatureSet=self.FeatureSet)
        Attack.train(*self.shadowDataSyn)

        guesses = Attack.attack(self.synthetic)
        self.assertEqual(len(guesses), len(self.synthetic))

        ## Test linkage
        Attack.train(*self.shadowDataSan)
        guesses = Attack.attack(self.sanitised, attemptLinkage=True, target=self.target)
        self.assertEqual(len(guesses), len(self.sanitised))

    def test_mia_svc(self):
        print('\nTest MIA SVC')
        Attack = MIAttackClassifierLinearSVC(metadata=self.metadata, FeatureSet=self.FeatureSet)
        Attack.train(*self.shadowDataSyn)

        guesses = Attack.attack(self.synthetic)
        self.assertEqual(len(guesses), len(self.synthetic))

        ## Test linkage
        Attack.train(*self.shadowDataSan)
        guesses = Attack.attack(self.sanitised, attemptLinkage=True, target=self.target)
        self.assertEqual(len(guesses), len(self.sanitised))






