from pandas import DataFrame

from generative_models.sdgym_utils.privbn import PrivBN
from generative_models.generative_model import GenerativeModel
from utils.constants import *



class PrivBaySDGym(GenerativeModel):
    """
    Differentially private Bayesian network based on the SDGym C++ implementation
    """

    def __init__(self, metadata, epsilon=1, theta=20, max_samples=25000):
        self.metadata = self._read_meta(metadata)
        self.privbn = PrivBN(epsilon=epsilon, theta=theta, max_samples=max_samples)

        self.trained = False
        self.datatype = DataFrame

        self.__name__ = f'PrivBaySDGymEps{epsilon}'

    def fit(self, data):
        self.privbn.fit(data, self.metadata)

        self.trained = True

    def generate_samples(self, nsamples):
        synthetic_data = self.privbn.sample(nsamples)

        return synthetic_data

    def _read_meta(self, metadata):
        """ Read metadata from metadata file."""
        metadict = {'fields': {}}

        for cdict in metadata['columns']:
            col = cdict['name']
            coltype = cdict['type']

            if coltype == FLOAT or coltype == INTEGER:
                metadict['fields'][col] = {'type': 'continuous'}

            elif coltype == CATEGORICAL or coltype == ORDINAL:
                metadict['fields'][col] = {'type': 'categorical'}

            else:
                raise ValueError(f'Unknown data type {coltype} for attribute {col}')

        return metadict