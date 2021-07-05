"""Generative models adapted from https://github.com/DataResponsibly/DataSynthesizer"""
# Copyright <2018> <dataresponsibly.com>

from numpy.random import seed, laplace, choice
from pandas import DataFrame, merge
from itertools import product

from generative_models.data_synthesiser_utils.datatypes.FloatAttribute import FloatAttribute
from generative_models.data_synthesiser_utils.datatypes.IntegerAttribute import IntegerAttribute
from generative_models.data_synthesiser_utils.datatypes.StringAttribute import StringAttribute
from generative_models.data_synthesiser_utils.utils import bayes_worker, normalize_given_distribution, exponential_mechanism

from generative_models.generative_model import GenerativeModel

from utils.constants import *
from utils.logging import LOGGER


class IndependentHistogram(GenerativeModel):

    def __init__(self, metadata, histogram_bins=10, infer_ranges=False, multiprocess=True):
        self.metadata = self._read_meta(metadata)
        self.histogram_bins = histogram_bins

        self.datatype = DataFrame
        self.multiprocess = bool(multiprocess)
        self.infer_ranges = bool(infer_ranges)

        self.DataDescriber = None

        self.trained = False

        self.__name__ = 'IndependentHistogram'

    def fit(self, data):
        assert isinstance(data, self.datatype), f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(data)}'
        LOGGER.debug(f'Start fitting IndependentHistogram model to data of shape {data.shape}...')
        if self.trained:
            self.trained = False
            self.DataDescriber = None

        self.DataDescriber = DataDescriber(self.metadata, self.histogram_bins, self.infer_ranges)
        self.DataDescriber.describe(data)
        LOGGER.debug(f'Finished fitting IndependentHistogram')
        self.trained = True

    def generate_samples(self, nsamples):
        assert self.trained, "Model must be fitted to some data first"

        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')
        synthetic_dataset = DataFrame(columns=self.DataDescriber.attr_names)
        for attr_name, Attr in self.DataDescriber.attr_dict.items():
            binning_indices = Attr.sample_binning_indices_in_independent_attribute_mode(nsamples)
            synthetic_dataset[attr_name] = Attr.sample_values_from_binning_indices(binning_indices)

        LOGGER.debug(f'Generated synthetic dataset of size {nsamples}')
        return synthetic_dataset

    def _read_meta(self, metadata):
        """ Read metadata from metadata file."""
        metadict = {}

        for cdict in metadata['columns']:
            col = cdict['name']
            coltype = cdict['type']

            if coltype == FLOAT or coltype == INTEGER:
                metadict[col] = {
                    'type': coltype,
                    'min': cdict['min'],
                    'max': cdict['max']
                }

            elif coltype == CATEGORICAL or coltype == ORDINAL:
                metadict[col] = {
                    'type': coltype,
                    'categories': cdict['i2s'],
                    'size': len(cdict['i2s'])
                }

            else:
                raise ValueError(f'Unknown data type {coltype} for attribute {col}')

        return metadict


class BayesianNet(GenerativeModel):
    """
    A BayesianNet model using non-private GreedyBayes to learn conditional probabilities
    """
    def __init__(self, metadata, histogram_bins=10, degree=1, infer_ranges=False, multiprocess=True, seed=None):
        self.metadata = self._read_meta(metadata)
        self.histogram_bins = histogram_bins
        self.degree = degree
        self.num_attributes = len(metadata['columns'])

        self.multiprocess = bool(multiprocess)
        self.infer_ranges = bool(infer_ranges)
        self.seed = seed
        self.datatype = DataFrame

        self.bayesian_network = None
        self.conditional_probabilities = None
        self.DataDescriber = None
        self.trained = False

        self.__name__ = 'BayesianNet'

    def fit(self, data):
        assert isinstance(data, self.datatype), f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(data)}'
        assert len(list(data)) >= 2, "BayesianNet requires at least 2 attributes(i.e., columns) in dataset."
        LOGGER.debug(f'Start training BayesianNet on data of shape {data.shape}...')
        if self.trained:
            self.trained = False
            self.DataDescriber = None
            self.bayesian_network = None
            self.conditional_probabilities = None

        self.DataDescriber = DataDescriber(self.metadata, self.histogram_bins, self.infer_ranges)
        self.DataDescriber.describe(data)

        encoded_df = DataFrame(columns=self.DataDescriber.attr_names)
        for attr_name, column in self.DataDescriber.attr_dict.items():
            encoded_df[attr_name] = column.encode_values_into_bin_idx()

        self.bayesian_network = self._greedy_bayes_linear(encoded_df, self.degree)

        self.conditional_probabilities = self._construct_conditional_probabilities(self.bayesian_network, encoded_df)

        LOGGER.debug(f'Finished training Bayesian net')
        self.trained = True

    def generate_samples(self, nsamples):
        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')
        assert self.trained, "Model must be fitted to some real data first"
        synthetic_data = DataFrame(columns=self.DataDescriber.attr_names)

        # Get samples for attributes modelled in Bayesian net
        encoded_dataset = self._generate_encoded_dataset(nsamples)

        for attr in self.DataDescriber.attr_names:
            column = self.DataDescriber.attr_dict[attr]
            if attr in encoded_dataset:
                synthetic_data[attr] = column.sample_values_from_binning_indices(encoded_dataset[attr])
            else:
                # For attributes not in BN use independent attribute mode
                binning_indices = column.sample_binning_indices_in_independent_attribute_mode(nsamples)
                synthetic_data[attr] = column.sample_values_from_binning_indices(binning_indices)

        return synthetic_data

    def _generate_encoded_dataset(self, nsamples):
        encoded_df = DataFrame(columns=self._get_sampling_order(self.bayesian_network))

        bn_root_attr = self.bayesian_network[0][1][0]
        root_attr_dist = self.conditional_probabilities[bn_root_attr]
        encoded_df[bn_root_attr] = choice(len(root_attr_dist), size=nsamples, p=root_attr_dist)

        for child, parents in self.bayesian_network:
            child_conditional_distributions = self.conditional_probabilities[child]

            for parents_instance in child_conditional_distributions.keys():
                dist = child_conditional_distributions[parents_instance]
                parents_instance = list(eval(parents_instance))

                filter_condition = ''
                for parent, value in zip(parents, parents_instance):
                    filter_condition += f"(encoded_df['{parent}']=={value})&"

                filter_condition = eval(filter_condition[:-1])
                size = encoded_df[filter_condition].shape[0]
                if size:
                    encoded_df.loc[filter_condition, child] = choice(len(dist), size=size, p=dist)

            # Fill any nan values by sampling from marginal child distribution
            marginal_dist = self.DataDescriber.attr_dict[child].distribution_probabilities
            null_idx = encoded_df[child].isnull()
            encoded_df.loc[null_idx, child] = choice(len(marginal_dist), size=null_idx.sum(), p=marginal_dist)

        encoded_df[encoded_df.columns] = encoded_df[encoded_df.columns].astype(int)

        return encoded_df

    def _get_sampling_order(self, bayesian_net):
        order = [bayesian_net[0][1][0]]
        for child, _ in bayesian_net:
            order.append(child)
        return order

    def _greedy_bayes_linear(self, encoded_df, k=1):
        """Construct a Bayesian Network (BN) using greedy algorithm."""
        dataset = encoded_df.astype(str, copy=False)

        # Optional: Fix sed for reproducibility
        if self.seed is not None:
            seed(self.seed)

        root_attribute = choice(dataset.columns)
        V = [root_attribute]
        rest_attributes = set(dataset.columns)
        rest_attributes.remove(root_attribute)
        bayesian_net = []
        while rest_attributes:
            parents_pair_list = []
            mutual_info_list = []

            num_parents = min(len(V), k)
            for child, split in product(rest_attributes, range(len(V) - num_parents + 1)):
                task = (child, V, num_parents, split, dataset)
                res = bayes_worker(task)
                parents_pair_list += res[0]
                mutual_info_list += res[1]

            idx = mutual_info_list.index(max(mutual_info_list))

            bayesian_net.append(parents_pair_list[idx])
            adding_attribute = parents_pair_list[idx][0]
            V.append(adding_attribute)
            rest_attributes.remove(adding_attribute)

        return bayesian_net

    def _construct_conditional_probabilities(self, bayesian_network, encoded_dataset):
        k = len(bayesian_network[-1][1])
        conditional_distributions = {}

        # first k+1 attributes
        root = bayesian_network[0][1][0]
        kplus1_attributes = [root]
        for child, _ in bayesian_network[:k]:
            kplus1_attributes.append(child)

        freqs_of_kplus1_attributes = self._get_attribute_frequency_counts(kplus1_attributes, encoded_dataset)

        # get distribution of root attribute
        root_marginal_freqs = freqs_of_kplus1_attributes.loc[:, [root, 'count']].groupby(root).sum()['count']
        conditional_distributions[root] = normalize_given_distribution(root_marginal_freqs).tolist()

        for idx, (child, parents) in enumerate(bayesian_network):
            conditional_distributions[child] = {}

            if idx < k:
                stats = freqs_of_kplus1_attributes.copy().loc[:, parents + [child, 'count']]
            else:
                stats = self._get_attribute_frequency_counts(parents + [child], encoded_dataset)

            stats = DataFrame(stats.loc[:, parents + [child, 'count']].groupby(parents + [child]).sum())

            if len(parents) == 1:
                for parent_instance in stats.index.levels[0]:
                    dist = normalize_given_distribution(stats.loc[parent_instance]['count']).tolist()
                    conditional_distributions[child][str([parent_instance])] = dist
            else:
                for parents_instance in product(*stats.index.levels[:-1]):
                    dist = normalize_given_distribution(stats.loc[parents_instance]['count']).tolist()
                    conditional_distributions[child][str(list(parents_instance))] = dist

        return conditional_distributions

    def _get_attribute_frequency_counts(self, attributes, encoded_dataset):
        # Get attribute counts for category combinations present in data
        counts = encoded_dataset.groupby(attributes).size()
        counts.name = 'count'
        counts = counts.reset_index()

        # Get all possible attribute combinations
        attr_combs = [range(self.DataDescriber.attr_dict[attr].domain_size) for attr in attributes]
        full_space = DataFrame(columns=attributes, data=list(product(*attr_combs)))
        # stats.reset_index(inplace=True)
        full_counts = merge(full_space, counts, how='left')
        full_counts.fillna(0, inplace=True)

        return full_counts

    def _read_meta(self, metadata):
        """ Read metadata from metadata file."""
        metadict = {}

        for cdict in metadata['columns']:
            col = cdict['name']
            coltype = cdict['type']

            if coltype == FLOAT or coltype == INTEGER:
                metadict[col] = {
                    'type': coltype,
                    'min': cdict['min'],
                    'max': cdict['max']
                }

            elif coltype == CATEGORICAL or coltype == ORDINAL:
                metadict[col] = {
                    'type': coltype,
                    'categories': cdict['i2s'],
                    'size': len(cdict['i2s'])
                }

            else:
                raise ValueError(f'Unknown data type {coltype} for attribute {col}')

        return metadict


class PrivBayes(BayesianNet):
    """"
    A differentially private BayesianNet model using GreedyBayes
    """
    def __init__(self, metadata, histogram_bins=10, degree=1, epsilon=.1, infer_ranges=False, multiprocess=True, seed=None):
        super().__init__(metadata=metadata, histogram_bins=histogram_bins, degree=degree, infer_ranges=infer_ranges, multiprocess=multiprocess, seed=seed)

        self.epsilon = float(epsilon)

        self.__name__ = f'PrivBayesEps{self.epsilon}'

    @property
    def laplace_noise_scale(self):
        return 2 * (self.num_attributes - self.degree) / (self.epsilon / 2)

    def _greedy_bayes_linear(self, encoded_df, k=1):
        """Construct a Bayesian Network (BN) using greedy algorithm."""
        dataset = encoded_df.astype(str, copy=False)
        num_tuples, num_attributes = dataset.shape

        # Optional: Fix seed for reproducibility
        if self.seed is not None:
            seed(self.seed)

        attr_to_is_binary = {attr: dataset[attr].unique().size <= 2 for attr in dataset}

        root_attribute = choice(dataset.columns)
        V = [root_attribute]
        rest_attributes = set(dataset.columns)
        rest_attributes.remove(root_attribute)
        bayesian_net = []
        while rest_attributes:
            parents_pair_list = []
            mutual_info_list = []

            num_parents = min(len(V), k)
            for child, split in product(rest_attributes, range(len(V) - num_parents + 1)):
                task = (child, V, num_parents, split, dataset)
                res = bayes_worker(task)
                parents_pair_list += res[0]
                mutual_info_list += res[1]

            sampling_distribution = exponential_mechanism(self.epsilon/2, mutual_info_list, parents_pair_list, attr_to_is_binary,
                                                          num_tuples, num_attributes)
            idx = choice(list(range(len(mutual_info_list))), p=sampling_distribution)

            bayesian_net.append(parents_pair_list[idx])
            adding_attribute = parents_pair_list[idx][0]
            V.append(adding_attribute)
            rest_attributes.remove(adding_attribute)

        return bayesian_net

    def _get_attribute_frequency_counts(self, attributes, encoded_dataset):
        """ Differentially private mechanism to get attribute frequency counts"""
        # Get attribute counts for category combinations present in data
        counts = encoded_dataset.groupby(attributes).size()
        counts.name = 'count'
        counts = counts.reset_index()

        # Get all possible attribute combinations
        attr_combs = [range(self.DataDescriber.attr_dict[attr].domain_size) for attr in attributes]
        full_space = DataFrame(columns=attributes, data=list(product(*attr_combs)))
        full_counts = merge(full_space, counts, how='left')
        full_counts.fillna(0, inplace=True)

        # Get Laplace noise sample
        noise_sample = laplace(0, scale=self.laplace_noise_scale, size=full_counts.index.size)
        full_counts['count'] += noise_sample
        full_counts.loc[full_counts['count'] < 0, 'count'] = 0

        return full_counts


class DataDescriber(object):
    def __init__(self, metadata, histogram_bins, infer_ranges=False):
        self.metadata = metadata
        self.histogram_bins = histogram_bins
        self.infer_ranges = infer_ranges

        self.attr_dict = None
        self.attr_names = None

    def describe(self, df):
        self.attr_names = self._get_attr_names()
        self.attr_dict = self._represent_input_dataset_by_columns(df)

        for col, Attribute in self.attr_dict.items():
            Attribute.infer_distribution()

    def _get_attr_names(self):
        return [c for c in self.metadata.keys()]

    def _represent_input_dataset_by_columns(self, df):
        attr_dict = {}

        for col, cdict in self.metadata.items():
            coltype = cdict['type']

            paras = (col, df[col], self.histogram_bins)
            if coltype in NUMERICAL:
                if coltype == FLOAT:
                    Attribute = FloatAttribute(*paras)
                else:
                    Attribute = IntegerAttribute(*paras)

                if self.infer_ranges:
                    cmin, cmax = min(df[col]), max(df[col])
                else:
                    cmin, cmax = cdict['min'], cdict['max']

                Attribute.set_domain(domain=(cmin, cmax))

            elif coltype in STRINGS:
                Attribute = StringAttribute(*paras)
                if self.infer_ranges:
                    ccats = list(df[col].unique())
                else:
                    ccats = cdict['categories']

                Attribute.set_domain(domain=ccats)

            else:
                raise Exception(f'The DataType of {col} is unknown.')

            attr_dict[col] = Attribute

        return attr_dict







