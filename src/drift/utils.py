import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
import sys


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


def safe_isinstance(obj, class_path_str):
    # this function is copy-paste from the code of the SHAP Python library
    # Copyright (c) 2018 Scott Lundberg

    """
    Acts as a safe version of isinstance without having to explicitly
    import packages which may not exist in the users environment.
    Checks if obj is an instance of type specified by class_path_str.
    Parameters
    ----------
    obj: Any
        Some object you want to test against
    class_path_str: str or list
        A string or list of strings specifying full class paths
        Example: `sklearn.ensemble.RandomForestRegressor`
    Returns
    --------
    bool: True if isinstance is true and the package exists, False otherwise
    """
    if isinstance(class_path_str, str):
        class_path_strs = [class_path_str]
    elif isinstance(class_path_str, list) or isinstance(class_path_str, tuple):
        class_path_strs = class_path_str
    else:
        class_path_strs = ['']

    # try each module path in order
    for class_path_str in class_path_strs:
        if "." not in class_path_str:
            raise ValueError("class_path_str must be a string or list of strings specifying a full \
                module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'")

        # Splits on last occurence of "."
        module_name, class_name = class_path_str.rsplit(".", 1)

        # here we don't check further if the model is not imported, since we shouldn't have
        # an object of that types passed to us if the model the type is from has never been
        # imported. (and we don't want to import lots of new modules for no reason)
        if module_name not in sys.modules:
            continue

        module = sys.modules[module_name]

        #Get class
        _class = getattr(module, class_name, None)

        if _class is None:
            continue

        if isinstance(obj, _class):
            return True

    return False
