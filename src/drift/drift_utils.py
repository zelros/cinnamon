import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, wasserstein_distance, ks_2samp
import matplotlib.pyplot as plt


def compute_distribution_cat(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None,
                             max_n_cat: int = None):
    if sample_weights1 is None:
        sample_weights1 = np.ones_like(a1)
    if sample_weights2 is None:
        sample_weights2 = np.ones_like(a2)

    # compute cat_map is needed
    def _compute_cat_map(a: np.array, sample_weights: np.array, max_n_cat: float):
        # sort categories by size
        unique_cat = np.unique(a).tolist()
        total_weight = np.sum(sample_weights)
        cat_weights = [np.sum(sample_weights[a == cat]) / total_weight for cat in unique_cat]
        sorted_cat = [cat for _, cat in sorted(zip(cat_weights, unique_cat), reverse=True)]

        # create category mapping to reducce number of categories
        cat_map = {}
        for i, cat in enumerate(sorted_cat):
            if i < (max_n_cat-1):
                cat_map[cat] = cat
            else:
                cat_map[cat] = 'other_cat_agg'
        return cat_map

    if max_n_cat is not None:
        cat_map = _compute_cat_map(np.concatenate((a1, a2)), np.concatenate((sample_weights1, sample_weights2)),
                                   max_n_cat)
    else:
        cat_map = None

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


def chi2_test(a: np.array, b: np.array):
    contingency_table = pd.crosstab(a, b)
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    return {'chi2_stat': chi2_stat,
            'p_value': p_value,
            'dof': dof,
            'contingency_table': contingency_table}


def compute_drift_num(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None):
    # TODO: does ks generalize to weighted samples ?
    if (sample_weights1 is None and sample_weights2 is None or
            np.all(sample_weights1 == sample_weights1[0]) and np.all(sample_weights2 == sample_weights2[0])):
        kolmogorov_smirnov = ks_2samp(a1, a2)
    else:
        kolmogorov_smirnov = None
    return {'mean_difference': compute_mean_diff(a1, a2, sample_weights1, sample_weights2),
            'wasserstein': wasserstein_distance(a1, a2, sample_weights1, sample_weights2),
            'kolmogorov_smirnov': kolmogorov_smirnov}


def compute_drift_cat(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None):
    if (sample_weights1 is None and sample_weights2 is None or
            np.all(sample_weights1 == sample_weights1[0]) and np.all(sample_weights2 == sample_weights2[0])):
        # TODO: does chi2 generalize to weighted samples ?
        # indeed, I am sure it is not sufficient to compute the contingency table with weights. So the chi2 formula need
        # to take weights into account
        # chi2 should not take max_n_cat into account. If pbm with number of cat, should be handled by
        # chi2_test internally with a proper solution

        # TODO chi2 not working for now
        #chi2 = chi2_test(np.concatenate((a1, a2)), np.array([0] * len(a1) + [1] * len(a2)))
        chi2 = None
    else:
        chi2 = None
    return {'wasserstein': wasserstein_distance_for_cat(a1, a2, sample_weights1, sample_weights2),
            'chi2_test': chi2}


def plot_drift_cat(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None, title=None,
                   max_n_cat: float = None):

    # compute both distributions
    distrib = compute_distribution_cat(a1, a2, sample_weights1, sample_weights2, max_n_cat)
    bar_height = np.array([v for v in distrib.values()]) # len(distrib) rows and 2 columns

    #plot
    index = np.arange(len(distrib))
    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(index, bar_height[:, 0], bar_width, label="Dataset 1")
    ax.bar(index+bar_width, bar_height[:, 1], bar_width, label="Dataset 2")

    ax.set_xlabel('Category')
    ax.set_ylabel('Percentage')
    ax.set_title(title)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(list(distrib.keys()), rotation=30)
    ax.legend()
    plt.show()


def plot_drift_num(a1: np.array, a2: np.array, sample_weights1: np.array=None, sample_weights2: np.array=None,
                   title=None):
    #distrib = compute_distribution_num(a1, a2, sample_weights1, sample_weights2)
    plt.hist(a1, bins=100, density=True, weights=sample_weights1, alpha=0.3)
    plt.hist(a2, bins=100, density=True, weights=sample_weights2, alpha=0.3)
    plt.legend(['Dataset 1', 'Dataset 2'])
    plt.title(title)
    plt.show()


def compute_distribution_num(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None):
    pass
