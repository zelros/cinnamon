import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def compute_distribution_cat(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None):
    if sample_weights1 is None:
        sample_weights1 = np.ones_like(a1)
    if sample_weights2 is None:
        sample_weights2 = np.ones_like(a2)
    unique_cat1 = np.unique(a1).tolist()
    unique_cat2 = np.unique(a2).tolist()
    unique_cat = unique_cat1 + [cat for cat in unique_cat2 if cat not in unique_cat1]
    total_weight1 = np.sum(sample_weights1)
    total_weight2 = np.sum(sample_weights2)
    distrib = {}
    for cat in unique_cat:
        distrib[cat] = [np.sum(sample_weights1[a1 == cat]) / total_weight1,
                        np.sum(sample_weights2[a2 == cat]) / total_weight2]
    return distrib


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
