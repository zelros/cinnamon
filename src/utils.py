import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def compute_distribution_cat(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None,
                             min_cat_weight: float = None):
    if sample_weights1 is None:
        sample_weights1 = np.ones_like(a1)
    if sample_weights2 is None:
        sample_weights2 = np.ones_like(a2)

    # compute cat_map is needed
    def _compute_cat_map(a: np.array, sample_weights: np.array, min_cat_weight: float):
        unique_cat = np.unique(a).tolist()
        total_weight = np.sum(sample_weights)
        cat_map = {}
        for cat in unique_cat:
            if np.sum(sample_weights[a == cat]) / total_weight < min_cat_weight:
                cat_map[cat] = f'under_{min_cat_weight * 100}%_agg'
            else:
                cat_map[cat] = cat
        return cat_map

    if min_cat_weight is not None:
        cat_map = _compute_cat_map(np.concatenate((a1, a2)), np.concatenate((sample_weights1, sample_weights2)),
                                   min_cat_weight)
    else:
        cat_map=None

    # compute the distribution
    unique_cat1 = np.unique(a1).tolist()
    unique_cat2 = np.unique(a2).tolist()
    unique_cat = unique_cat1 + [cat for cat in unique_cat2 if cat not in unique_cat1]
    total_weight1 = np.sum(sample_weights1)
    total_weight2 = np.sum(sample_weights2)
    if cat_map is not None:
        distrib = {cat: [0, 0] for cat in cat_map.values()}
        for cat in unique_cat:
            distrib[cat_map[cat]] = [distrib[cat_map[cat]][0] + np.sum(sample_weights1[a1 == cat]) / total_weight1,
                                     distrib[cat_map[cat]][1] + np.sum(sample_weights2[a2 == cat]) / total_weight2]
    else:
        distrib = {}
        for cat in unique_cat:
            distrib[cat] = [np.sum(sample_weights1[a1 == cat]) / total_weight1,
                            np.sum(sample_weights2[a2 == cat]) / total_weight2]

    return distrib


def compute_mean_diff(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None):
    if sample_weights1 is None:
        sample_weights1 = np.ones_like(a1)
    if sample_weights2 is None:
        sample_weights2 = np.ones_like(a2)
        mean1 = np.sum(a1 * sample_weights1) / np.sum(sample_weights1)
        mean2 = np.sum(a2 * sample_weights2) / np.sum(sample_weights2)
    return mean2 - mean1


def wasserstein_distance_for_cat(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None):
    # this correspond to wasserstein distance where we assume a distance 1 between two categories of the feature
    distrib = compute_distribution_cat(a1, a2, sample_weights1, sample_weights2)
    drift = 0
    for cat in distrib.keys():
        drift += abs(distrib[cat][0] - distrib[cat][1]) / 2
    return drift


def softmax(a: np.array):
    """
    :param a: array of log_softmax
    :return:
    """
    return np.exp(a) / np.sum(np.exp(a), axis=1)[:, None]


def chi2_test(a: np.array, b: np.array):
    contingency_table = pd.crosstab(a, b)
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    return {'chi2_stat': chi2_stat,
            'p_value': p_value,
            'dof': dof,
            'contingency_table': contingency_table}
