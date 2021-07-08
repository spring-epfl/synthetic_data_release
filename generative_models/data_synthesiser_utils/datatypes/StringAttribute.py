import numpy as np

from generative_models.data_synthesiser_utils.datatypes.AbstractAttribute import AbstractAttribute
from generative_models.data_synthesiser_utils.datatypes.utils.DataType import DataType
from generative_models.data_synthesiser_utils.utils import normalize_given_distribution, generate_random_string


class StringAttribute(AbstractAttribute):
    """Variable min and max are the lengths of the shortest and longest strings.

    """

    def __init__(self, name, data, histogram_size):
        super().__init__(name, data, histogram_size)
        self.is_categorical = True
        self.is_numerical = False
        self.data_type = DataType.STRING
        self.data_dropna_len = self.data_dropna.astype(str).map(len)

    def set_domain(self, domain=None):
        if domain is not None:
            lengths = [len(i) for i in domain]
            self.min = min(lengths)
            self.max = max(lengths)
            self.distribution_bins = np.array(domain)
        else:
            self.min = int(self.data_dropna_len.min())
            self.max = int(self.data_dropna_len.max())
            self.distribution_bins = self.data_dropna.unique()

        self.domain_size = len(self.distribution_bins)

    def infer_distribution(self):

        histogram = self.data_dropna.value_counts()
        for attr_cat in set(self.distribution_bins) - set(histogram.index):
            histogram[attr_cat] = 0
        histogram = histogram[self.distribution_bins]
        self.distribution_probabilities = normalize_given_distribution(histogram)

    def generate_values_as_candidate_key(self, n):
        length = np.random.randint(self.min, self.max)
        vectorized = np.vectorize(lambda x: '{}{}'.format(generate_random_string(length), x))
        return vectorized(np.arange(n))

    def sample_values_from_binning_indices(self, binning_indices):
        return super().sample_values_from_binning_indices(binning_indices)
