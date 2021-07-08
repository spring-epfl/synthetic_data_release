from os import path
from sklearn.mixture import GaussianMixture
from generative_models.generative_model import GenerativeModel

import logging
from logging.config import fileConfig
dirname = path.dirname(__file__)
logconfig = path.join(dirname, '../logging_config.ini')
fileConfig(logconfig)
logger = logging.getLogger(__name__)

class GaussianMixtureModel(GenerativeModel):

    def __init__(self):
        self.gm = GaussianMixture()
        self.trained = False

    def fit(self, data):
        """Fit a gaussian mixture model to the input data. Input data is assumed to be of shape (n_samples, n_features)
        See https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture.fit for details"""
        logger.debug(f'Start fitting GaussianMixtureModel to data of shape {data.shape}...')
        self.gm.fit(data)
        logger.debug(f'Finished fitting GMM')
        self.trained = True

    def generate_samples(self, nsamples):
        """Generate random samples from the fitted Gaussian distribution"""
        assert self.trained, "Model must first be fitted to some data."
        logger.debug(f'Generate synthetic dataset of size {nsamples}')
        synthetic_data, _ = self.gm.sample(nsamples)
        return synthetic_data

