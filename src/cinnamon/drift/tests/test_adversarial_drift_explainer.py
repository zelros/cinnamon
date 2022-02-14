import pandas as pd, numpy as np
from numpy.testing import assert_allclose
from sklearn import datasets
from sklearn.model_selection import train_test_split
from cinnamon.common.stat_utils import BaseStatisticalTestResult, Chi2TestResult
from cinnamon.drift import AdversarialDriftExplainer
from cinnamon.drift.drift_utils import (DriftMetricsCat, DriftMetricsNum,
                                        assert_drift_metrics_equal,
                                        assert_drift_metrics_list_equal)
from ...common.constants import NUMPY_atol

RANDOM_SEED = 2021


def test_adversarial_drift_explainer():
    dataset = datasets.load_iris()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target).map({0: 'Iris-Setosa', 1: 'Iris-Versicolour', 2: 'Iris-Virginica'})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
    drift_explainer = AdversarialDriftExplainer(seed=RANDOM_SEED, verbosity=False).fit(X_train, X_test, y_train, y_test)

    # adversarial_drift_values
    assert_allclose(drift_explainer.get_adversarial_drift_values(),
                    np.array([[0.27580956],
                              [0.3008618],
                              [0.18539894],
                              [0.23792973]]),
                    atol=NUMPY_atol)

    # adversarial_correction_weights
    assert_allclose(drift_explainer.get_adversarial_correction_weights(),
                    np.array([0.90810081, 0.76698897, 0.99673733, 1.05017385, 1.26964731,
                              0.76698897, 1.00623148, 1.11323871, 1.26964731, 0.95846368,
                              1.28322436, 0.95846368, 0.93051401, 0.93051401, 0.69659946,
                              0.69659946, 0.83154643, 0.69659946, 0.95846368, 0.95846368,
                              1.11323871, 1.05017385, 1.26964731, 1.26964731, 0.90810081,
                              1.05017385, 0.83154643, 0.69659946, 0.85965477, 0.99673733,
                              1.26964731, 0.69659946, 0.69659946, 1.11323871, 1.00623148,
                              1.00979752, 0.95846368, 1.26964731, 1.11323871, 1.01535569,
                              1.13832889, 0.76698897, 1.01535569, 1.00979752, 1.11323871,
                              1.26964731, 1.01535569, 0.83154643, 1.26964731, 1.11323871,
                              1.13832889, 0.95846368, 0.69659946, 0.95846368, 0.76698897,
                              0.90810081, 1.26964731, 0.95846368, 1.16355731, 1.26964731,
                              1.00623148, 0.85965477, 1.05017385, 0.98884336, 1.00623148,
                              0.93051401, 0.93051401, 1.26964731, 1.00623148, 1.26964731,
                              1.11323871, 1.28322436, 0.93051401, 0.99673733, 1.26964731,
                              0.69659946, 0.76698897, 1.26964731, 0.76698897, 1.00623148,
                              0.83154643, 0.99673733, 1.26246993, 0.90810081, 1.13832889,
                              1.05017385, 1.13832889, 1.11323871, 0.93051401, 1.05017385,
                              0.69659946, 0.90810081, 1.00623148, 0.76698897, 1.11323871,
                              1.05017385, 1.00623148, 0.95592598, 1.00623148, 0.90810081,
                              0.69659946, 1.11323871, 1.05017385, 1.01535569, 0.99673733]),
                    atol=NUMPY_atol)

    # target drift
    assert_drift_metrics_equal(drift_explainer.get_target_drift(),
                               DriftMetricsCat(wasserstein=0.09523809523809523,
                                               jensen_shannon=0.07382902143706498,
                                               chi2_test=Chi2TestResult(
                                                   statistic=1.3333333333333333,
                                                   pvalue=0.5134171190325922,
                                                   dof=2,
                                                   contingency_table=pd.DataFrame(
                                                       [[33.0, 34.0, 38.0],
                                                        [17.0, 16.0, 12.0]],
                                                       index=['X1', 'X2'],
                                                       columns=['Iris-Setosa',
                                                                'Iris-Versicolour',
                                                                'Iris-Virginica']))))

    # feature_drifts
    feature_drifts_ref = [DriftMetricsNum(mean_difference=-0.18571428571428505, wasserstein=0.19968253968253974,
                                          ks_test=BaseStatisticalTestResult(statistic=0.16507936507936508,
                                                                            pvalue=0.3237613427576299)),
                          DriftMetricsNum(mean_difference=-0.08825396825396803, wasserstein=0.1301587301587301,
                                          ks_test=BaseStatisticalTestResult(statistic=0.14285714285714285,
                                                                            pvalue=0.499646880472137)),
                          DriftMetricsNum(mean_difference=-0.2765079365079357, wasserstein=0.2777777777777778,
                                          ks_test=BaseStatisticalTestResult(statistic=0.1523809523809524,
                                                                            pvalue=0.41885114043708227)),
                          DriftMetricsNum(mean_difference=-0.16412698412698457, wasserstein=0.16412698412698412,
                                          ks_test=BaseStatisticalTestResult(statistic=0.17142857142857143,
                                                                            pvalue=0.2821678346768163))]
    assert_drift_metrics_list_equal(drift_explainer.get_feature_drifts(),
                                    feature_drifts_ref)

    assert_drift_metrics_equal(drift_explainer.get_feature_drift('sepal length (cm)'),
                               DriftMetricsNum(mean_difference=-0.18571428571428505,
                                               wasserstein=0.19968253968253974,
                                               ks_test=BaseStatisticalTestResult(statistic=0.16507936507936508,
                                                                                 pvalue=0.3237613427576299)))

    assert_drift_metrics_equal(drift_explainer.get_feature_drift(2),
                               DriftMetricsNum(mean_difference=-0.2765079365079357,
                                               wasserstein=0.2777777777777778,
                                               ks_test=BaseStatisticalTestResult(statistic=0.1523809523809524,
                                                                                 pvalue=0.41885114043708227)))

    assert drift_explainer.feature_names == ['sepal length (cm)',
                                             'sepal width (cm)',
                                             'petal length (cm)',
                                             'petal width (cm)']

    assert drift_explainer.feature_subset == ['sepal length (cm)',
                                              'sepal width (cm)',
                                              'petal length (cm)',
                                              'petal width (cm)']

    assert drift_explainer.n_features == 4

    assert drift_explainer.task == 'classification'

    # sample_weights1
    assert_allclose(drift_explainer.sample_weights1[:5],
                    np.array([1., 1., 1., 1., 1.]),
                    atol=NUMPY_atol)

    # sample_weights2
    assert_allclose(drift_explainer.sample_weights2[:5],
                    np.array([1., 1., 1., 1., 1.]),
                    atol=NUMPY_atol)
