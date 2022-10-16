import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import wasserstein_distance, ks_2samp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Union
from ..common.stat_utils import (compute_distributions_cat,
                                 compute_distributions_num,
                                 compute_mean_diff,
                                 wasserstein_distance_for_cat,
                                 jensen_shannon_distance,
                                 chi2_test,
                                 ks_weighted,
                                 BaseStatisticalTestResult,
                                 Chi2TestResult,
                                 PerformanceMetrics,
                                 )
from ..common.constants import FLOAT_atol, ModelAgnosticDriftValueType
from ..common.math_utils import threshold_array, divide_with_0, map_array_to_bins


def compute_drift_num(a1: np.array, a2: np.array, sample_weights1: np.array = None, sample_weights2: np.array = None,
                      js_bins: int = 10, js_base=None):
    if (sample_weights1 is None and sample_weights2 is None or
            np.all(sample_weights1 == sample_weights1[0]) and np.all(sample_weights2 == sample_weights2[0])):
        ks_test_object = ks_2samp(a1, a2)
        ks_test = BaseStatisticalTestResult(statistic=ks_test_object.statistic,
                                            pvalue=ks_test_object.pvalue)
    else:
        # 'ks_weighted' return a dictionnary with the good format
        ks_test = ks_weighted(a1, a2, sample_weights1, sample_weights2)
    return DriftMetricsNum(mean_difference=compute_mean_diff(a1, a2, sample_weights1, sample_weights2),
                           wasserstein=wasserstein_distance(a1, a2, sample_weights1, sample_weights2),
                           ks_test=ks_test)


def compute_drift_cat(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None, js_base=None):
    return DriftMetricsCat(wasserstein=wasserstein_distance_for_cat(a1, a2, sample_weights1, sample_weights2),
                           jensen_shannon=jensen_shannon_distance(a1, a2, js_base, sample_weights1, sample_weights2),
                           chi2_test=chi2_test(a1, a2, sample_weights1, sample_weights2))


def plot_drift_cat(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None, title=None,
                   max_n_cat: int = None, figsize=(10, 6),
                   legend_labels: Tuple[str, str] = ('Dataset 1', 'Dataset 2')) -> None:
    # compute both distributions
    distrib = compute_distributions_cat(a1, a2, sample_weights1, sample_weights2, max_n_cat)
    bar_height = np.array([v for v in distrib.values()])  # len(distrib) rows and 2 columns

    # plot
    index = np.arange(len(distrib))
    bar_width = 0.35
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(index, bar_height[:, 0], bar_width)
    ax.bar(index + bar_width, bar_height[:, 1], bar_width)

    ax.set_xlabel('Category')
    ax.set_ylabel('Percentage')
    ax.set_title(title)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(list(distrib.keys()), rotation=30)
    ax.legend(legend_labels)
    plt.show()


def plot_drift_num(a1: np.array, a2: np.array, sample_weights1: np.array = None, sample_weights2: np.array = None,
                   title=None, figsize=(7, 5), bins=10,
                   legend_labels: Tuple[str, str] = ('Dataset 1', 'Dataset 2')) -> None:
    distribs = compute_distributions_num(a1, a2, bins, sample_weights1, sample_weights2)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(a1, bins=distribs['bin_edges'], density=True, weights=sample_weights1, alpha=0.3)
    ax.hist(a2, bins=distribs['bin_edges'], density=True, weights=sample_weights2, alpha=0.3)
    ax.legend(legend_labels)
    plt.title(title)
    plt.show()


@dataclass
class AbstractDriftMetrics:
    def assert_equal(self, other) -> None:
        pass


@dataclass
class DriftMetricsNum(AbstractDriftMetrics):
    mean_difference: float
    wasserstein: float
    ks_test: BaseStatisticalTestResult

    def assert_equal(self, other) -> None:
        assert isinstance(other, DriftMetricsNum)
        assert_allclose(self.mean_difference, other.mean_difference, atol=FLOAT_atol)
        assert_allclose(self.wasserstein, other.wasserstein, atol=FLOAT_atol)
        self.ks_test.assert_equal(other.ks_test)


@dataclass
class DriftMetricsCat(AbstractDriftMetrics):
    wasserstein: float
    jensen_shannon: float
    chi2_test: Chi2TestResult

    def assert_equal(self, other) -> None:
        assert isinstance(other, DriftMetricsCat)
        assert_allclose(self.wasserstein, other.wasserstein, atol=FLOAT_atol)
        assert_allclose(self.jensen_shannon, other.jensen_shannon, atol=FLOAT_atol)
        self.chi2_test.assert_equal(other.chi2_test)


def assert_drift_metrics_equal(drift_metrics1: AbstractDriftMetrics,
                               drift_metrics2: AbstractDriftMetrics) -> None:
    drift_metrics1.assert_equal(drift_metrics2)


def assert_drift_metrics_list_equal(l1: List[AbstractDriftMetrics], l2: List[AbstractDriftMetrics]) -> None:
    assert len(l1) == len(l2)
    for i in range(len(l1)):
        assert_drift_metrics_equal(l1[i], l2[i])


@dataclass
class PerformanceMetricsDrift:
    dataset1: PerformanceMetrics
    dataset2: PerformanceMetrics


def assert_performance_metrics_drift_equal(performance_metrics_drift1: PerformanceMetricsDrift,
                                           performance_metrics_drift2: PerformanceMetricsDrift):
    performance_metrics_drift1.dataset1.assert_equal(performance_metrics_drift2.dataset1)
    performance_metrics_drift1.dataset2.assert_equal(performance_metrics_drift2.dataset2)


def compute_model_agnostic_drift_value_num(a1: np.array, a2: np.array, type:str, sample_weights1: np.array,
                                     sample_weights2: np.array, predictions1: np.array,
                                     predictions2: np.array, bins, max_ratio: float) -> List[float]:
    '''
    symmetrical version of compute_agnostic_drift_value_num_one_way
    '''
    # 2 maps 1 means distribution of a2 is mapped to distribution of a1
    drift_importance_2_maps_1 = compute_model_agnostic_drift_value_num_one_way(a1, a2, type, sample_weights1, sample_weights2,
                                             predictions2, bins, max_ratio)
    drift_importance_1_maps_2 = compute_model_agnostic_drift_value_num_one_way(a2, a1, type, sample_weights2, sample_weights1,
                                             predictions1, bins, max_ratio)
    if type == ModelAgnosticDriftValueType.WASSERSTEIN.value:
        return [(x + y)/2 for x, y in zip(drift_importance_2_maps_1, drift_importance_1_maps_2)]
    else:
        return [(-x + y)/2 for x, y in zip(drift_importance_2_maps_1, drift_importance_1_maps_2)]
    

def compute_model_agnostic_drift_value_num_one_way(a1: np.array, a2: np.array, type:str, sample_weights1: np.array,
                                             sample_weights2: np.array, predictions2: np.array,
                                             bins, max_ratio: float) -> List[float]:
    """
    only wasserstein distance used right now but could be mean, kolmogorov ?
    """
    distributions = compute_distributions_num(a1, a2, bins, sample_weights1, sample_weights2)
    normalized_hist1 = distributions['hist1'] / np.sum(distributions['hist1'])
    normalized_hist2 = distributions['hist2'] / np.sum(distributions['hist2'])
    ratio2 = divide_with_0(normalized_hist1, normalized_hist2)
    thresholded_ratio2 = threshold_array(ratio2, max_ratio=max_ratio)
    bins_ids = map_array_to_bins(a2, distributions['bin_edges'])
    correction_weights2 = thresholded_ratio2[bins_ids]

    return compute_model_agnostic_drift_value_from_weights(predictions2, type, sample_weights2, correction_weights2)


def compute_model_agnostic_drift_value_cat(a1: np.array, a2: np.array, type: str, sample_weights1: np.array,
                                     sample_weights2: np.array, predictions1: np.array,
                                     predictions2: np.array, max_ratio: float, max_n_cat: int) -> List[float]:
    '''
    symmetrical version of compute_model_agnostic_drift_value_cat_one_way
    '''
    # 2 maps 1 means distribution of a2 is mapped to distribution of a1
    drift_importance_2_maps_1 = compute_model_agnostic_drift_value_cat_one_way(a1, a2, type, sample_weights1, sample_weights2,
                                             predictions2, max_ratio, max_n_cat)
    drift_importance_1_maps_2 = compute_model_agnostic_drift_value_cat_one_way(a2, a1, type, sample_weights2, sample_weights1,
                                             predictions1, max_ratio, max_n_cat)
    if type == ModelAgnosticDriftValueType.WASSERSTEIN.value:
        return [(x + y)/2 for x, y in zip(drift_importance_2_maps_1, drift_importance_1_maps_2)]
    else:
        return [(-x + y)/2 for x, y in zip(drift_importance_2_maps_1, drift_importance_1_maps_2)]


def compute_model_agnostic_drift_value_cat_one_way(a1: np.array, a2: np.array, type: str, sample_weights1: np.array,
                                             sample_weights2: np.array, predictions2: np.array,
                                             max_ratio: float, max_n_cat: int) -> List[float]:
    distributions, category_map = compute_distributions_cat(a1, a2, sample_weights1, sample_weights2, max_n_cat=max_n_cat, return_category_map=True)
    #plot_drift_cat(a1=a1, a2=a2, sample_weights1=sample_weights1, sample_weights2=sample_weights2)
    ratio = np.array([probas[0] / probas[1] if probas[1] > 0 else 0 for probas in distributions.values()])
    thresholded_ratio = threshold_array(ratio, max_ratio=max_ratio)
    thresholded_ratio_dict = {k: thresholded_ratio[i] for i, k in enumerate(distributions.keys())}
    if category_map:
        a2_mapped = np.array(list(map(lambda x: category_map[x], a2)))
        correction_weights = np.array(list(map(lambda x: thresholded_ratio_dict[x], a2_mapped)))
    else:
        correction_weights = np.array(list(map(lambda x: thresholded_ratio_dict[x], a2)))
    
    return compute_model_agnostic_drift_value_from_weights(predictions2, type, sample_weights2, correction_weights)


def compute_model_agnostic_drift_value_from_weights(predictions: np.array, type: str, sample_weights: np.array,
                                                    correction_weights: np.array) -> List[float]:
    drift_importances = []
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    for j in range(predictions.shape[1]):
        if type == ModelAgnosticDriftValueType.WASSERSTEIN.value:
            # wasserstein distance between distrib of predictions with original weights, and 
            # distrib of predictions with corrected weights
            drift_importance = wasserstein_distance(predictions[:, j], predictions[:, j],
                                                    sample_weights, sample_weights * correction_weights)
        if type == ModelAgnosticDriftValueType.MEAN.value:
            drift_importance = compute_mean_diff(predictions[:, j], predictions[:, j],
                                                 sample_weights, sample_weights * correction_weights)
        drift_importances.append(drift_importance)
    return drift_importances
