"""
Utils for generative models adapted from https://github.com/DataResponsibly/DataSynthesizer

Copyright <2018> <dataresponsibly.com>

Licensed under MIT License
"""

from math import log, ceil
from numpy import array, exp, isinf, full_like
from numpy.random import choice, laplace
from string import ascii_lowercase
from itertools import combinations, product
from pandas import Series, DataFrame, merge
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score


def mutual_information(labels_x: Series, labels_y: DataFrame):
    """Mutual information of distributions in format of Series or DataFrame.

    Parameters
    ----------
    labels_x : Series
    labels_y : DataFrame
    """
    if labels_y.shape[1] == 1:
        labels_y = labels_y.iloc[:, 0]
    else:
        labels_y = labels_y.apply(lambda x: ' '.join(x.values), axis=1)

    return mutual_info_score(labels_x, labels_y)


def pairwise_attributes_mutual_information(dataset):
    """Compute normalized mutual information for all pairwise attributes. Return a DataFrame."""
    sorted_columns = sorted(dataset.columns)
    mi_df = DataFrame(columns=sorted_columns, index=sorted_columns, dtype=float)
    for row in mi_df.columns:
        for col in mi_df.columns:
            mi_df.loc[row, col] = normalized_mutual_info_score(dataset[row].astype(str),
                                                               dataset[col].astype(str),
                                                               average_method='arithmetic')
    return mi_df


def normalize_given_distribution(frequencies):
    distribution = array(frequencies, dtype=float)
    distribution = distribution.clip(0)  # replace negative values with 0
    summation = distribution.sum()
    if summation > 0:
        if isinf(summation):
            return normalize_given_distribution(isinf(distribution))
        else:
            return distribution / summation
    else:
        return full_like(distribution, 1 / distribution.size)


def infer_numerical_attributes_in_dataframe(dataframe):
    describe = dataframe.describe()
    # DataFrame.describe() usually returns 8 rows.
    if describe.shape[0] == 8:
        return set(describe.columns)
    # DataFrame.describe() returns less than 8 rows when there is no numerical attribute.
    else:
        return set()


def display_bayesian_network(bn):
    length = 0
    for child, _ in bn:
        if len(child) > length:
            length = len(child)

    print('Constructed Bayesian network:')
    for child, parents in bn:
        print("    {0:{width}} has parents {1}.".format(child, parents, width=length))


def generate_random_string(length):
    return ''.join(choice(list(ascii_lowercase), size=length))


def bayes_worker(paras):
    child, V, num_parents, split, dataset = paras
    parents_pair_list = []
    mutual_info_list = []

    if split + num_parents - 1 < len(V):
        for other_parents in combinations(V[split + 1:], num_parents - 1):
            parents = list(other_parents)
            parents.append(V[split])
            parents_pair_list.append((child, parents))
            mi = mutual_information(dataset[child], dataset[parents])
            mutual_info_list.append(mi)

    return parents_pair_list, mutual_info_list


def get_encoded_attribute_distribution(attributes, encoded_dataset):
    data = encoded_dataset.copy().loc[:, attributes]
    data['count'] = 1
    stats = data.groupby(attributes).sum()

    iterables = [range(int(encoded_dataset[attr].max()) + 1) for attr in attributes]
    full_space = DataFrame(columns=attributes, data=list(product(*iterables)))
    stats.reset_index(inplace=True)
    stats = merge(full_space, stats, how='left')
    stats.fillna(0, inplace=True)

    return stats


def get_noisy_distribution_of_attributes(attributes, encoded_dataset, epsilon=0.1):
    data = encoded_dataset.copy().loc[:, attributes]
    data['count'] = 1
    stats = data.groupby(attributes).sum()

    iterables = [range(int(encoded_dataset[attr].max()) + 1) for attr in attributes]
    full_space = DataFrame(columns=attributes, data=list(product(*iterables)))
    stats.reset_index(inplace=True)
    stats = merge(full_space, stats, how='left')
    stats.fillna(0, inplace=True)

    if epsilon:
        k = len(attributes) - 1
        num_tuples, num_attributes = encoded_dataset.shape
        noise_para = laplace_noise_parameter(k, num_attributes, num_tuples, epsilon)
        laplace_noises = laplace(0, scale=noise_para, size=stats.index.size)
        stats['count'] += laplace_noises
        stats.loc[stats['count'] < 0, 'count'] = 0

    return stats


def laplace_noise_parameter(k, num_attributes, num_tuples, epsilon):
    """The noises injected into conditional distributions. PrivBayes Algorithm 1."""
    return 2 * (num_attributes - k) / (num_tuples * epsilon)


def calculate_sensitivity(num_tuples, child, parents, attr_to_is_binary):
    """Sensitivity function for Bayesian network construction. PrivBayes Lemma 1.
    Parameters
    ----------
    num_tuples : int
        Number of tuples in sensitive dataset.
    Return
    --------
    int
        Sensitivity value.
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


def calculate_delta(num_attributes, sensitivity, epsilon):
    """Computing delta, which is a factor when applying differential privacy.
    More info is in PrivBayes Section 4.2 "A First-Cut Solution".
    Parameters
    ----------
    num_attributes : int
        Number of attributes in dataset.
    sensitivity : float
        Sensitivity of removing one tuple.
    epsilon : float
        Parameter of differential privacy.
    """
    return (num_attributes - 1) * sensitivity / epsilon


def exponential_mechanism(epsilon, mutual_info_list, parents_pair_list, attr_to_is_binary, num_tuples, num_attributes):
    """Applied in Exponential Mechanism to sample outcomes."""
    delta_array = []
    for (child, parents) in parents_pair_list:
        sensitivity = calculate_sensitivity(num_tuples, child, parents, attr_to_is_binary)
        delta = calculate_delta(num_attributes, sensitivity, epsilon)
        delta_array.append(delta)

    mi_array = array(mutual_info_list) / (2 * array(delta_array))
    mi_array = exp(mi_array)
    mi_array = normalize_given_distribution(mi_array)
    return mi_array