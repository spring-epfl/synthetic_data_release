"""
Generative model training algorithm based on the CTGANSynthesiser published by
Xu et al., 2019: "Modeling Tabular data using Conditional GAN"

Dependencies: CTGAN <https://github.com/sdv-dev/CTGAN>
"""

from os import path
from pandas import DataFrame

from .generative_model import GenerativeModel
from ctgan import CTGANSynthesizer

import torch
from torch.multiprocessing.pool import Pool
# Change forking method
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

from synthetic_data.utils.logging import LOGGER


class CTGAN(GenerativeModel):
    """
    A generative adversarial network for tabular data
    """

    def __init__(self, metadata,
                 embedding_dim=128,
                 gen_dim=(256, 256),
                 dis_dim=(256, 256),
                 l2scale=1e-6,
                 batch_size=500,
                 epochs=300):

        self.synthesiser = CTGANSynthesizer(embedding_dim,
                                            gen_dim,
                                            dis_dim,
                                            l2scale,
                                            batch_size,
                                            epochs)
        self.metadata = metadata
        self.datatype = DataFrame

        self.trained = False

        self.__name__ = 'CTGAN'

    def fit(self, rawTrain):
        """
        Fit a generative model of the training data distribution.
        See <https://github.com/sdv-dev/CTGAN> for details.

        :param rawTrain: DataFrame or ndarray: Training set
        """
        assert isinstance(rawTrain, self.datatype), f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(rawTrain)}'

        LOGGER.debug(f'Start fitting {self.__class__.__name__} to data of shape {rawTrain.shape}...')
        self.synthesiser.fit(rawTrain, self.metadata)

        LOGGER.debug(f'Finished fitting')
        self.trained = True

    def generate_samples(self, nsamples):
        """
        Samples synthetic data records from the fitted generative distribution

        :param nsamples: int: Number of synthetic records to generate
        :return: synData: DataFrame: A synthetic dataset
        """
        assert self.trained, "Model must first be fitted to some data."

        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')
        synData = self.synthesiser.sample(nsamples)

        return synData
