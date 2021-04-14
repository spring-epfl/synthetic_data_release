"""
Generative models adapted from https://github.com/DataResponsibly/DataSynthesizer

Copyright <2018> <dataresponsibly.com>

Licensed under MIT License
"""
from math import log
from numpy import array, exp
from numpy.random import seed, laplace, choice
from pandas import DataFrame, merge
from itertools import product

import torch
from torch.multiprocessing import Pool

# Change forking method
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

from .data_synthesiser_utils.datatypes.FloatAttribute import FloatAttribute
from .data_synthesiser_utils.datatypes.IntegerAttribute import IntegerAttribute
from .data_synthesiser_utils.datatypes.StringAttribute import StringAttribute
from .data_synthesiser_utils.utils import bayes_worker, normalize_given_distribution
from .generative_model import GenerativeModel

from synthetic_data.utils.logging import LOGGER
from synthetic_data.utils.constants import *

PROCESSES = 16


class IndependentHistogram(GenerativeModel):
    """ A generative model that approximates the joint data distribution as a set of independent marginals """

    def __init__(self, metadata, histogram_bins=10):
        """

        :param category_threshold: int: Threshold for classifying an attribute as categorical rather as unique ID
        :param histogram_bins: int: Number of bins for binning continuous attributes
        """

        self.metadata = metadata
        self.histogram_bins = histogram_bins
        self.data_describer = None
        self.datatype = DataFrame

        self.trained = False

        self.__name__ = 'IndependentHistogram'

    def fit(self, rawTrain):
        """
        Fit a generative model of the training data distribution.
        The IndHist model extracts frequency counts from each attribute independently.

        :param rawTrain: DataFrame: Training dataset
        :return: None
        """
        assert isinstance(rawTrain, self.datatype), f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(rawTrain)}'

        if self.trained:
            # Make sure to delete previous data description
            self.trained = False
            self.data_describer = None

        LOGGER.debug(f'Start fitting IndependentHistogram model to data of shape {rawTrain.shape}...')
        self.data_describer = DataDescriber(self.metadata, self.histogram_bins)
        self.data_describer.describe(rawTrain)
        LOGGER.debug(f'Finished fitting IndependentHistogram')

        self.trained = True

    def generate_samples(self, nsamples):
        """
        Samples synthetic data records from the fitted generative distribution

        :param nsamples: int: Number of synthetic records to generate
        :return: synthetic_dataset: DataFrame: A synthetic dataset
        """
        assert self.trained, "Model must be fitted to some raw data first"

        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')
        synthetic_dataset = DataFrame(columns=self.data_describer.metadata['attribute_list'])
        for attr_name, column in self.data_describer.attr_dict.items():
            binning_indices = column.sample_binning_indices_in_independent_attribute_mode(nsamples)
            synthetic_dataset[attr_name] = column.sample_values_from_binning_indices(binning_indices)

        LOGGER.debug(f'Generated synthetic dataset of size {nsamples}')

        return synthetic_dataset


class BayesianNet(GenerativeModel):
    """
    A BayesianNet model using non-private GreedyBayes to learn conditional probabilities
    """
    def __init__(self, metadata, histogram_bins=10, k=1, multiprocess=False, seed=None):
        """

        :param metadata: dict:  A dicitionary that contains column metadata
        :param histogram_bins: int: Number of bins for binning continuous attributes
        :param k: int: Maximum degree of the nodes in the Bayesian network
        :param multiprocess: bool: If set to TRUE will parallelise training
        :param seed: int: Seed random choice for root attribute
        """
        self.metadata = metadata
        self.histogram_bins = histogram_bins
        self.k = k
        self.num_attributes = len(metadata['columns'])

        self.multiprocess = multiprocess
        self.seed = seed
        self.datatype = DataFrame

        self.bayesian_network = None
        self.conditional_probabilities = None
        self.data_describer = None
        self.trained = False

        self.__name__ = 'BayesianNet'

    def fit(self, rawTrain):
        """
        Fit a generative model of the training data distribution.
        The BayNet model first models the conditional independence structure of data attributes
        as a Bayesian network and then fits a set of conditional marginals to the training data.

        :param rawTrain: DataFrame: Training dataset
        """
        assert isinstance(rawTrain, self.datatype), f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(rawTrain)}'

        if self.trained:
            self.trained = False
            self.data_describer = None
            self.bayesian_network = None
            self.conditional_probabilities = None

        LOGGER.debug(f'Start training BayesianNet on data of shape {rawTrain.shape}...')
        self.data_describer  = DataDescriber(self.metadata, self.histogram_bins)
        self.data_describer.describe(rawTrain)

        encoded_df = DataFrame()
        for attr_name, column in self.data_describer.attr_dict.items():
            encoded_df[attr_name] = column.encode_values_into_bin_idx()
        if encoded_df.shape[1] < 2:
            raise Exception("BayesianNet requires at least 2 attributes(i.e., columns) in dataset.")

        if self.multiprocess:
            self.bayesian_network = self._greedy_bayes_multiprocess(encoded_df, self.k)
        else:
            self.bayesian_network = self._greedy_bayes_linear(encoded_df, self.k)

        self.conditional_probabilities = self._construct_conditional_probabilities(self.bayesian_network, encoded_df)

        LOGGER.debug(f'Finished training Bayesian net')
        self.trained = True

    def generate_samples(self, nsamples):
        """
        Samples synthetic data records from the fitted generative distribution

        :param nsamples: int: Number of synthetic records to generate
        :return: synthetic_dataset: DataFrame: A synthetic dataset
        """
        assert self.trained, "Model must be fitted to some real data first"

        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')

        all_attributes = self.data_describer.metadata['attribute_list']
        synthetic_data = DataFrame(columns=all_attributes)

        # Get samples for attributes modelled in Bayesian net
        encoded_dataset = self._generate_encoded_dataset(nsamples)

        for attr in all_attributes:
            column = self.data_describer.attr_dict[attr]
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
            marginal_dist = self.data_describer.attr_dict[child].distribution_probabilities
            null_idx = encoded_df[child].isnull()
            encoded_df.loc[null_idx, child] = choice(len(marginal_dist), size=null_idx.sum(), p=marginal_dist)

        encoded_df[encoded_df.columns] = encoded_df[encoded_df.columns].astype(int)
        return encoded_df

    def _get_sampling_order(self, bayesian_net):
        order = [bayesian_net[0][1][0]]
        for child, _ in bayesian_net:
            order.append(child)
        return order

    def _greedy_bayes_multiprocess(self, encoded_df, k=1):
        """Construct a Bayesian Network (BN) using greedy algorithm."""
        dataset = encoded_df.astype(str, copy=False)

        root_attribute = choice(dataset.columns)
        V = [root_attribute]
        rest_attributes = set(dataset.columns)
        rest_attributes.remove(root_attribute)
        bayesian_net = []
        while rest_attributes:
            parents_pair_list = []
            mutual_info_list = []

            num_parents = min(len(V), k)
            tasks = [(child, V, num_parents, split, dataset) for child, split in
                     product(rest_attributes, range(len(V) - num_parents + 1))]
            with Pool(processes=PROCESSES) as pool:
                res_list = pool.map(bayes_worker, tasks)

            for res in res_list:
                parents_pair_list += res[0]
                mutual_info_list += res[1]

            idx = mutual_info_list.index(max(mutual_info_list))

            bayesian_net.append(parents_pair_list[idx])
            adding_attribute = parents_pair_list[idx][0]
            V.append(adding_attribute)
            rest_attributes.remove(adding_attribute)

        return bayesian_net

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
        """Construct marginal conditional attribute distributions."""
        k = len(bayesian_network[-1][1])
        conditional_distributions = {}

        # Get conditional frequency counts for first k+1 attributes
        root = bayesian_network[0][1][0]
        kplus1_attributes = [root]
        for child, _ in bayesian_network[:k]:
            kplus1_attributes.append(child)

        freqs_of_kplus1_attributes = self._get_attribute_frequency_counts(kplus1_attributes, encoded_dataset)

        # Get marginal distribution of root attribute
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
        """ Extract conditional frequency counts from dataset."""
        # Get attribute counts for category combinations present in data
        counts = encoded_dataset.groupby(attributes).size()
        counts.name = 'count'
        counts = counts.reset_index()

        # Get all possible attribute combinations
        attr_combs = [range(self.data_describer.attr_dict[attr].domain_size) for attr in attributes]
        full_space = DataFrame(columns=attributes, data=list(product(*attr_combs)))
        # stats.reset_index(inplace=True)
        full_counts = merge(full_space, counts, how='left')
        full_counts.fillna(0, inplace=True)

        return full_counts


class PrivBayes(BayesianNet):
    """"
    A differentially private BayesianNet model using GreedyBayes
    """
    def __init__(self, metadata, histogram_bins=10, k=1, epsilon=.1, multiprocess=False, seed=None):
        """

        :param metadata: dict:  Dictionary that contains column metadata
        :param histogram_bins: int: Number of bins for binning continuous attributes
        :param k: int: Maximum degree of the nodes in the Bayesian network
        :param eps: float: Privacy parameter
        :param multiprocess: bool: If set to TRUE will parallelise training
        :param seed: int: Seed randmoness for root attribute choice
        """

        super().__init__(metadata=metadata, histogram_bins=histogram_bins, k=k, multiprocess=multiprocess)

        self.epsilon = epsilon

        self.__name__ = f'PrivBayesEps{self.epsilon}'

    @property
    def laplace_noise_scale(self):
        return 2 * (self.num_attributes - self.k) / (self.epsilon/2)

    def _greedy_bayes_multiprocess(self, encoded_df, k=1):
        """Construct a Bayesian Network using a differentially private version of the greedy Bayes algorithm."""
        dataset = encoded_df.astype(str, copy=False)
        num_tuples, _ = dataset.shape

        # Optional: Fix sed for reproducibility
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
            tasks = [(child, V, num_parents, split, dataset) for child, split in
                     product(rest_attributes, range(len(V) - num_parents + 1))]
            with Pool(processes=PROCESSES) as pool:
                res_list = pool.map(bayes_worker, tasks)

            for res in res_list:
                parents_pair_list += res[0]
                mutual_info_list += res[1]

            sampling_distribution = self._exponential_mechanism(mutual_info_list, parents_pair_list, attr_to_is_binary, num_tuples)
            idx = choice(list(range(len(mutual_info_list))), p=sampling_distribution)

            bayesian_net.append(parents_pair_list[idx])
            adding_attribute = parents_pair_list[idx][0]
            V.append(adding_attribute)
            rest_attributes.remove(adding_attribute)

        return bayesian_net

    def _greedy_bayes_linear(self, encoded_df, k=1):
        """Construct a Bayesian Network (BN) using greedy algorithm."""
        dataset = encoded_df.astype(str, copy=False)
        num_tuples, num_attributes = dataset.shape

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

            sampling_distribution = self._exponential_mechanism(mutual_info_list, parents_pair_list, attr_to_is_binary, num_tuples)
            idx = choice(list(range(len(mutual_info_list))), p=sampling_distribution)

            bayesian_net.append(parents_pair_list[idx])
            adding_attribute = parents_pair_list[idx][0]
            V.append(adding_attribute)
            rest_attributes.remove(adding_attribute)

        return bayesian_net

    def _exponential_mechanism(self, mutual_info_list, parents_pair_list, attr_to_is_binary, num_tuples):
        """Exponential mechanism to construct differentially private attribute choice."""
        delta_array = []
        for (child, parents) in parents_pair_list:
            sensitivity = self.calculate_sensitivity_exp_mech(num_tuples, child, parents, attr_to_is_binary)
            delta = self._get_delta(sensitivity)
            delta_array.append(delta)

        mutual_info_array = array(mutual_info_list) / (2 * array(delta_array))
        mutual_info_array = exp(mutual_info_array)
        mutual_info_array = normalize_given_distribution(mutual_info_array)

        return mutual_info_array

    @staticmethod
    def calculate_sensitivity_exp_mech(num_tuples, child, parents, attr_to_is_binary):
        """Sensitivity function for Bayesian network construction. PrivBayes Lemma 1.
        :param num_tuples: int: Number of tuples in sensitive dataset
        :param child: str: Child attribute to add to the network
        :param parents: list: List of potential parent attributes
        :param attr_to_is_binary: dict: Dict with info abou which attributes are binary

        :return int: Sensitivity of the query
        """
        if attr_to_is_binary[child] or (len(parents) == 1 and attr_to_is_binary[parents[0]]):
            a = log(num_tuples) / num_tuples
            b = (num_tuples - 1) / num_tuples
            b_inv = num_tuples / (num_tuples - 1)
            return a + b * log(b_inv)
        else:
            a = (2 / num_tuples) * log((num_tuples + 1) / 2)
            b = (1 - 1 / num_tuples) * log(1 + 2 / (num_tuples - 1))
            return a + b

    def _get_delta(self, sensitivity):
        return (self.num_attributes - 1) * sensitivity / (self.epsilon/2)

    def _get_attribute_frequency_counts(self, attributes, encoded_dataset):
        """ Differentially private mechanism to get attribute frequency counts"""
        # Get attribute counts for category combinations present in data
        counts = encoded_dataset.groupby(attributes).size()
        counts.name = 'count'
        counts = counts.reset_index()

        # Get all possible attribute combinations
        attr_combs = [range(self.data_describer.attr_dict[attr].domain_size) for attr in attributes]
        full_space = DataFrame(columns=attributes, data=list(product(*attr_combs)))
        full_counts = merge(full_space, counts, how='left')
        full_counts.fillna(0, inplace=True)

        # Get Laplace noise sample
        noise_sample = laplace(0, scale=self.laplace_noise_scale, size=full_counts.index.size)
        full_counts['count'] += noise_sample
        full_counts.loc[full_counts['count'] < 0, 'count'] = 0

        return full_counts


class DataDescriber(object):
    def __init__(self, metadata, histogram_bins):
        self.metadata = metadata
        self.histogram_bins = histogram_bins

        self.attributes = {}

    def describe(self, df):
        self.attr_dict = self.represent_input_dataset_by_columns(df)

        attr_list = []
        for attr_name, column in self.attr_dict.items():
            attr_list.append(attr_name)
            column.infer_distribution()

        self.metadata['attribute_list'] = attr_list

    def represent_input_dataset_by_columns(self, df):
        attr_to_column = {}

        for cdict in self.metadata['columns']:
            data_type = cdict['type']
            attr_name = cdict['name']
            paras = (attr_name, df[attr_name], self.histogram_bins)

            if data_type == INTEGER:
                attr_to_column[attr_name] = IntegerAttribute(*paras)
                attr_to_column[attr_name].set_domain(domain=(cdict['min'], cdict['max']))
            elif data_type == FLOAT:
                attr_to_column[attr_name] = FloatAttribute(*paras)
                attr_to_column[attr_name].set_domain(domain=(cdict['min'], cdict['max']))
            elif data_type == CATEGORICAL or data_type == ORDINAL:
                attr_to_column[attr_name] = StringAttribute(*paras)
                attr_to_column[attr_name].set_domain(domain=cdict['i2s'])
            else:
                raise Exception(f'The DataType of {attr_name} is unknown.')

        return attr_to_column






