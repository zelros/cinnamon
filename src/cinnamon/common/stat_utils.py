import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, distributions
from sklearn.metrics import log_loss, mean_squared_error, explained_variance_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from dataclasses import dataclass
from pandas.testing import assert_frame_equal
from .math_utils import sigmoid, softmax
from typing import List
from scipy.spatial.distance import jensenshannon


# ---------------------------------------
#        Compute performance metrics ML
# ---------------------------------------

def compute_classification_metrics(y_true: np.array, y_pred: np.array, sample_weights: np.array,
                                   class_names: List[str], prediction_type: str = 'proba') -> dict:
    ohe_y_true = OneHotEncoder(categories=[class_names]).fit_transform(y_true.reshape(-1, 1))

    # case prediction_type in ['raw', 'proba']
    if prediction_type in ['raw', 'proba']:
        if prediction_type == 'raw':  # predictions are logit (binary classif) or log softmax (multiclass classif):
            if y_pred.ndim == 1:  # binary classif
                y_pred_proba = sigmoid(y_pred)
            else:  # y_pred.ndim > 1
                y_pred_proba = softmax(y_pred)
        else:  # prediction_type == 'proba
            y_pred_proba = y_pred
        return {'log_loss': log_loss(ohe_y_true, y_pred_proba, sample_weight=sample_weights)}

    else:  # case prediction_type == 'label'
        return {'accuracy': accuracy_score(y_true, y_pred, sample_weight=sample_weights)}


def compute_regression_metrics(y_true: np.array, y_pred: np.array, sample_weights: np.array) -> dict:
    # TODO: add more metrics here
    return {'mse': mean_squared_error(y_true, y_pred, sample_weight=sample_weights),
            'explained_variance': explained_variance_score(y_true, y_pred, sample_weight=sample_weights)}


# --------------------------------
#        Compute distribution
# --------------------------------

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

        # create category mapping to reduce number of categories
        cat_map = {}
        for i, cat in enumerate(sorted_cat):
            if i < (max_n_cat - 1):
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


def compute_distributions_num(a1: np.array, a2: np.array, bins, sample_weights1=None, sample_weights2=None):
    bin_edges = np.histogram_bin_edges(np.concatenate((a1, a2)), bins=bins)
    hist1 = np.histogram(a1, bins=bin_edges, weights=sample_weights1, density=False)[0]
    hist2 = np.histogram(a2, bins=bin_edges, weights=sample_weights2, density=False)[0]
    return {'bin_edges': bin_edges, 'hist1': hist1, 'hist2': hist2}


# --------------------------------
#        Distances
# --------------------------------

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


def jensen_shannon_distance(a1: np.array, a2: np.array, base=None, sample_weights1=None, sample_weights2=None):
    distrib = compute_distribution_cat(a1, a2, sample_weights1, sample_weights2)
    p = [v[0] for v in distrib.values()]
    q = [v[1] for v in distrib.values()]
    return jensenshannon(p, q, base=base)


# --------------------------------
#        Statistical tests
# --------------------------------

def chi2_test(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None):
    # TODO: generalization of Chi2 for weights != np.ones is complicated (need verif)
    # chi2 do not take max_n_cat into account. If pbm with number of cat, should be handled by
    # chi2_test internally with a proper solution
    distrib = compute_distribution_cat(a1, a2, sample_weights1, sample_weights2)
    contingency_table = pd.DataFrame({cat: pd.Series({'X1': distrib[cat][0] * len(a1), 'X2': distrib[cat][1] * len(a2)})
                                      for cat in distrib.keys()})
    statistic, pvalue, dof, expected = chi2_contingency(contingency_table)
    return Chi2TestResult(statistic, pvalue, dof, contingency_table)


def ks_weighted(data1, data2, wei1, wei2, alternative='two-sided'):
    # kolmogorov smirnov test for weighted samples
    # taken from https://stackoverflow.com/questions/40044375/how-to-calculate-the-kolmogorov-smirnov-statistic-between-two-weighted-samples
    # see also: https://github.com/scipy/scipy/issues/12315
    # TODO: verify p-value computation is good
    ix1 = np.argsort(data1)
    ix2 = np.argsort(data2)
    data1 = data1[ix1]
    data2 = data2[ix2]
    wei1 = wei1[ix1]
    wei2 = wei2[ix2]
    data = np.concatenate([data1, data2])
    cwei1 = np.hstack([0, np.cumsum(wei1) / sum(wei1)])
    cwei2 = np.hstack([0, np.cumsum(wei2) / sum(wei2)])
    cdf1we = cwei1[np.searchsorted(data1, data, side='right')]
    cdf2we = cwei2[np.searchsorted(data2, data, side='right')]
    d = np.max(np.abs(cdf1we - cdf2we))
    # calculate p-value
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    m, n = sorted([float(n1), float(n2)], reverse=True)
    en = m * n / (m + n)
    if alternative == 'two-sided':
        prob = distributions.kstwo.sf(d, np.round(en))
    else:
        z = np.sqrt(en) * d
        # Use Hodges' suggested approximation Eqn 5.3
        # Requires m to be the larger of (n1, n2)
        expt = -2 * z ** 2 - 2 * z * (m + 2 * n) / np.sqrt(m * n * (m + n)) / 3.0
        prob = np.exp(expt)
    return BaseStatisticalTestResult(statistic=d, pvalue=prob)


# -----------------------------------
#        Statistical tests results
# -----------------------------------

@dataclass(frozen=True)
class BaseStatisticalTestResult:
    statistic: float
    pvalue: float

    def assert_equal(self, other):
        assert isinstance(other, BaseStatisticalTestResult)
        assert self.statistic == other.statistic
        assert self.pvalue == other.pvalue


@dataclass(frozen=True)
class Chi2TestResult(BaseStatisticalTestResult):
    dof: int = None
    contingency_table: pd.DataFrame = None

    ## see later if this functin is needed
    #    def __eq__(self, other):
    #        if not isinstance(other, Chi2TestResult):
    #            return False
    #        elif self.statistic != other.statistic or self.pvalue != other.pvalue or self.dof != other.dof:
    #            return False
    #        elif not self.contingency_table.equals(other.contingency_table):
    #            return False
    #        else:
    #            return True

    def assert_equal(self, other):
        assert isinstance(other, Chi2TestResult)
        assert self.statistic == other.statistic
        assert self.pvalue == other.pvalue
        assert self.dof == other.dof
        assert_frame_equal(self.contingency_table, other.contingency_table)
