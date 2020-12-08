"""
Generative models adapted from https://github.com/DataResponsibly/DataSynthesizer

Copyright <2018> <dataresponsibly.com>

Licensed under MIT License
"""

from numpy import array_equal
from os import path

import torch
from torch.multiprocessing import Pool

# Change forking method
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

from .data_synthesiser_utils.datatypes.constants import *
from .data_synthesiser_utils.datatypes.FloatAttribute import FloatAttribute
from .data_synthesiser_utils.datatypes.IntegerAttribute import IntegerAttribute
from .data_synthesiser_utils.datatypes.SocialSecurityNumberAttribute import SocialSecurityNumberAttribute
from .data_synthesiser_utils.datatypes.StringAttribute import StringAttribute
from .data_synthesiser_utils.datatypes.DateTimeAttribute import DateTimeAttribute
from .data_synthesiser_utils.datatypes.utils.AttributeLoader import parse_json
from .data_synthesiser_utils.utils import *
from .generative_model import GenerativeModel

from synthetic_data.utils.logging import LOGGER

PROCESSES = 16


class IndependentHistogram(GenerativeModel):
    """ A generative model that approximates the joint data distribution as a set of independent marginals """

    def __init__(self, category_threshold=10, histogram_bins=10):
        """

        :param category_threshold: int: Threshold for classifying an attribute as categorical rather as unique ID
        :param histogram_bins: int: Number of bins for binning continuous attributes
        """

        self.category_threshold = category_threshold
        self.histogram_bins = histogram_bins
        self.data_describer = DataDescriber(category_threshold, histogram_bins)
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
        LOGGER.debug(f'Start fitting IndependentHistogram model to data of shape {rawTrain.shape}...')
        if self.trained:
            # Make sure to delete previous data description
            self.data_describer = DataDescriber(self.category_threshold, self.histogram_bins)
        self.data_describer.describe(rawTrain)
        LOGGER.debug(f'Finished fitting IndependentHistogram')
        self.trained = True

    def generate_samples(self, nsamples):
        """
        Samples synthetic data records from the fitted generative distribution

        :param nsamples: int: Number of synthetic records to generate
        :return: synthetic_dataset: DataFrame: A synthetic dataset
        """
        assert self.trained, "Model must be fitted to some real data first"
        LOGGER.debug(f'Generate synthetic dataset of size {nsamples}')
        all_attributes = list(self.data_describer.metadata['attribute_list'])
        synthetic_dataset = DataFrame(columns=all_attributes)
        for attr in all_attributes:
            attr_info = self.data_describer.data_description[attr]
            column = parse_json(attr_info)
            binning_indices = column.sample_binning_indices_in_independent_attribute_mode(nsamples)
            synthetic_dataset[attr] = column.sample_values_from_binning_indices(binning_indices)
        LOGGER.debug(f'Generated synthetic dataset of size {nsamples}')
        return synthetic_dataset


class BayesianNet(GenerativeModel):
    """
    A BayesianNet model using non-private GreedyBayes to learn conditional probabilities
    """
    def __init__(self, category_threshold=5, histogram_bins=5, k=1, multiprocess=False):
        """

        :param category_threshold: int:  Threshold for classifying an attribute as categorical rather as unique ID
        :param histogram_bins: int: Number of bins for binning continuous attributes
        :param k: int: Maximum degree of the nodes in the Bayesian network
        :param multiprocess: bool: If set to TRUE will parallelise training
        """
        self.data_describer = DataDescriber(category_threshold, histogram_bins)

        self.k = k # maximum number of parents in Bayesian network
        self.bayesian_network = None
        self.conditional_probabilities = None
        self.multiprocess = multiprocess
        self.datatype = DataFrame

        self.trained = False

        self.__name__ = 'BayesianNet'

    def fit(self, rawTrain):
        """
        Fit a generative model of the training data distribution.
        The BayNet model first models the conditional independence structure of data attributes
        as a Bayesian network and then fits a set of conditional marginals to the training data.

        :param rawTrain: DataFrame: Training dataset
        :return: None
        """
        assert isinstance(rawTrain, self.datatype), f'{self.__class__.__name__} expects {self.datatype} as input data but got {type(rawTrain)}'
        LOGGER.debug(f'Start training BayesianNet on data of shape {rawTrain.shape}...')
        if self.trained:
            self.trained = False
            self.data_describer.data_description = {}
            self.bayesian_network = None
            self.conditional_probabilities = None

        self.data_describer.describe(rawTrain)

        encoded_df = DataFrame()
        for attr in self.data_describer.metadata['attribute_list_hist']:
            column = self.data_describer.attr_dict[attr]
            encoded_df[attr] = column.encode_values_into_bin_idx()
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
                unconditioned_distribution = self.data_describer.data_description[child]['distribution_probabilities']
                encoded_df.loc[encoded_df[child].isnull(), child] = choice(len(unconditioned_distribution),
                                                                              size=encoded_df[child].isnull().sum(),
                                                                              p=unconditioned_distribution)
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

        dist_of_kplus1_attributes = get_encoded_attribute_distribution(kplus1_attributes, encoded_dataset)

        # get distribution of root attribute
        root_stats = dist_of_kplus1_attributes.loc[:, [root, 'count']].groupby(root).sum()['count']
        conditional_distributions[root] = normalize_given_distribution(root_stats).tolist()

        for idx, (child, parents) in enumerate(bayesian_network):
            conditional_distributions[child] = {}

            if idx < k:
                stats = dist_of_kplus1_attributes.copy().loc[:, parents + [child, 'count']]
            else:
                stats = get_encoded_attribute_distribution(parents + [child], encoded_dataset)

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


class PrivBayes(BayesianNet):
    """"
    A differentially private BayesianNet model using GreedyBayes
    """
    def __init__(self, category_threshold=5, histogram_bins=5, k=1, epsilon=.1, multiprocess=False):
        """

        :param category_threshold: int:  Threshold for classifying an attribute as categorical rather as unique ID
        :param histogram_bins: int: Number of bins for binning continuous attributes
        :param k: int: Maximum degree of the nodes in the Bayesian network
        :param eps: float: Privacy parameter
        :param multiprocess: bool: If set to TRUE will parallelise training
        """

        super().__init__(category_threshold=category_threshold, histogram_bins=histogram_bins, k=k, multiprocess=multiprocess)

        self.epsilon = epsilon

        self.__name__ = f'PrivBayesEps{self.epsilon}'

    def _greedy_bayes_multiprocess(self, encoded_df, k=1):
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
            tasks = [(child, V, num_parents, split, dataset) for child, split in
                     product(rest_attributes, range(len(V) - num_parents + 1))]
            with Pool(processes=PROCESSES) as pool:
                res_list = pool.map(bayes_worker, tasks)

            for res in res_list:
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

            sampling_distribution = exponential_mechanism(self.epsilon/2, mutual_info_list, parents_pair_list, attr_to_is_binary,
                                                          num_tuples, num_attributes)
            idx = choice(list(range(len(mutual_info_list))), p=sampling_distribution)


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

        noisy_dist_of_kplus1_attributes = get_noisy_distribution_of_attributes(kplus1_attributes, encoded_dataset, self.epsilon/2)

        # generate noisy distribution of root attribute.
        root_stats = noisy_dist_of_kplus1_attributes.loc[:, [root, 'count']].groupby(root).sum()['count']
        conditional_distributions[root] = normalize_given_distribution(root_stats).tolist()

        for idx, (child, parents) in enumerate(bayesian_network):
            conditional_distributions[child] = {}

            if idx < k:
                stats = noisy_dist_of_kplus1_attributes.copy().loc[:, parents + [child, 'count']]
            else:
                stats = get_noisy_distribution_of_attributes(parents + [child], encoded_dataset, self.epsilon/2)

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


class DataDescriber(object):
    def __init__(self, category_threshold, histogram_bins):
        self.category_threshold = category_threshold
        self.histogram_bins = histogram_bins

        self.attributes = {}
        self.data_description = {}
        self.metadata = {}

    def describe(self, df):
        attr_to_datatype = self.infer_attribute_data_types(df)
        self.attr_dict = self.represent_input_dataset_by_columns(df, attr_to_datatype)

        for column in self.attr_dict.values():
            column.infer_domain()
            column.infer_distribution()
            self.data_description[column.name] = column.to_json()
        self.metadata = self.describe_metadata(df)

    def infer_attribute_data_types(self, df):
        attr_to_datatype = {}
        # infer data types
        numerical_attributes = infer_numerical_attributes_in_dataframe(df)

        for attr in list(df):
            column_dropna = df[attr].dropna()

            # Attribute is either Integer or Float.
            if attr in numerical_attributes:

                if array_equal(column_dropna, column_dropna.astype(int, copy=False)):
                    attr_to_datatype[attr] = INTEGER
                else:
                    attr_to_datatype[attr] = FLOAT

            # Attribute is either String or DateTime
            else:
                attr_to_datatype[attr] = STRING

        return attr_to_datatype

    def represent_input_dataset_by_columns(self, df, attr_to_datatype):
        attr_to_column = {}

        for attr in list(df):
            data_type = attr_to_datatype[attr]
            is_categorical = self.is_categorical(df[attr])
            paras = (attr, df[attr], self.histogram_bins, is_categorical)
            if data_type is INTEGER:
                attr_to_column[attr] = IntegerAttribute(*paras)
            elif data_type is FLOAT:
                attr_to_column[attr] = FloatAttribute(*paras)
            elif data_type is DATETIME:
                attr_to_column[attr] = DateTimeAttribute(*paras)
            elif data_type is STRING:
                attr_to_column[attr] = StringAttribute(*paras)
            elif data_type is SOCIAL_SECURITY_NUMBER:
                attr_to_column[attr] = SocialSecurityNumberAttribute(*paras)
            else:
                raise Exception(f'The DataType of {attr} is unknown.')

        return attr_to_column

    def describe_metadata(self, df):
        nrecords, nfeatures = df.shape
        all_attributes = list(df)
        hist_attributes = []
        str_attributes = []

        for attr in all_attributes:
            if attr in self.data_description.keys():
                column = self.data_description[attr]
                if column is STRING and not column.is_categorical:
                    str_attributes.append(attr)
                else:
                    hist_attributes.append(attr)

        metadata = {'num_records': nrecords,
                    'num_attributes': nfeatures,
                    'attribute_list': all_attributes,
                    'attribute_list_hist': hist_attributes,
                    'attribute_list_str': str_attributes}
        return metadata

    def is_categorical(self, data):
        """
        Detect whether an attribute is categorical.
        """
        return data.dropna().unique().size <= self.category_threshold







