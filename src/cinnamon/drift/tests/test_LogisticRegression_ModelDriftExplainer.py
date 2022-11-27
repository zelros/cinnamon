import pandas as pd
import numpy as np
from numpy.testing import assert_allclose
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostRegressor, CatBoostClassifier
from cinnamon.drift import ModelDriftExplainer
from cinnamon.drift.drift_utils import (DriftMetricsNum, DriftMetricsCat,
                                        PerformanceMetricsDrift,
                                        assert_drift_metrics_equal,
                                        assert_drift_metrics_list_equal,
                                        assert_performance_metrics_drift_equal)
from cinnamon.common.stat_utils import (BaseStatisticalTestResult, Chi2TestResult,
                                        RegressionMetrics, ClassificationMetrics)
from ...common.constants import NUMPY_atol

RANDOM_SEED = 2021


def test_LogisticRegression_ModelDriftExplainer():
    # load breast cancer data
    dataset = datasets.load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    # build logistic regression model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=2021)
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X=X_train, y=y_train)
    # fit ModelDriftExplainer
    drift_explainer = ModelDriftExplainer(clf, task='classification')
    drift_explainer.fit(X1=X_train, X2=X_test, y1=y_train, y2=y_test)

    # prediction drift "raw"
    prediction_drift_ref = [DriftMetricsNum(mean_difference=0.31398108070131237,
                                            wasserstein=0.6765664760436445,
                                            ks_test=BaseStatisticalTestResult(statistic=0.034015104763584, pvalue=0.9980870287496009))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(prediction_type='raw'),
                                    prediction_drift_ref, rtol=1e-2, atol=0.01)

    # prediction drift "proba"
    prediction_drift_proba_ref = [DriftMetricsNum(mean_difference=-0.0015880058846903244,
                                                  wasserstein=0.007597002373353593,
                                                  ks_test=BaseStatisticalTestResult(statistic=0.034015104763584, pvalue=0.9980870287496009))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(prediction_type='proba'),
                                    prediction_drift_proba_ref,
                                    rtol=1e-3, atol=0.01)

    # prediction drift "class"
    prediction_drift_class_ref = [DriftMetricsCat(wasserstein=0.010109024655440946,
                                                  jensen_shannon=0.0074381031914172785,
                                                  chi2_test=Chi2TestResult(statistic=0.018165244429467854,
                                                                           pvalue=0.8927870041515902,
                                                                           dof=1,
                                                                           contingency_table=pd.DataFrame([[146.0, 252.0], [61.0, 110.0]],
                                                                                                          index=['X1', 'X2'], columns=['0', '1'])))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(prediction_type='class'),
                                    prediction_drift_class_ref, rtol=1e-5, atol=1e5)

    # model agnostic drift importances "mean"
    assert_allclose(drift_explainer.get_model_agnostic_drift_values(type='mean'),
                    np.array([[0.19032076],
                              [-0.28289169],
                              [0.33290967],
                              [0.08994918],
                              [0.31350868],
                              [0.64245039],
                              [0.58636448],
                              [0.6953161],
                              [0.26833679],
                              [0.06572177],
                              [-0.08181331],
                              [0.28146249],
                              [-0.4876653],
                              [-0.45159551],
                              [-0.31079777],
                              [0.39828176],
                              [-0.20248079],
                              [-0.12408438],
                              [0.03461377],
                              [0.07748639],
                              [0.45247407],
                              [-0.06833666],
                              [0.32571906],
                              [0.48446051],
                              [0.57969521],
                              [0.34132056],
                              [0.73964219],
                              [0.43615057],
                              [0.39054499],
                              [-0.09303799]]),
                    atol=1e-2)

    # model agnostic drift values "wasserstein"
    assert_allclose(drift_explainer.get_model_agnostic_drift_values(type='wasserstein'),
                    np.array([[1.03235411],
                              [0.39986026],
                              [1.0088181],
                              [1.02632328],
                              [0.39491091],
                              [0.64523765],
                              [0.63177951],
                              [0.7021146],
                              [0.31187762],
                              [0.26989413],
                              [0.42054818],
                              [0.310376],
                              [0.60870494],
                              [0.61593346],
                              [0.3320135],
                              [0.41313262],
                              [0.2064267],
                              [0.20869742],
                              [0.2156971],
                              [0.12178143],
                              [0.93022366],
                              [0.21444828],
                              [1.09474119],
                              [0.94112299],
                              [0.62875716],
                              [0.37084863],
                              [0.73964219],
                              [0.48772089],
                              [0.47082898],
                              [0.35988305]]),
                    atol=1e-2)

    assert drift_explainer.task == 'classification'
    assert drift_explainer.class_names == ['0', '1']
    assert drift_explainer.feature_names == ['mean radius',
                                             'mean texture',
                                             'mean perimeter',
                                             'mean area',
                                             'mean smoothness',
                                             'mean compactness',
                                             'mean concavity',
                                             'mean concave points',
                                             'mean symmetry',
                                             'mean fractal dimension',
                                             'radius error',
                                             'texture error',
                                             'perimeter error',
                                             'area error',
                                             'smoothness error',
                                             'compactness error',
                                             'concavity error',
                                             'concave points error',
                                             'symmetry error',
                                             'fractal dimension error',
                                             'worst radius',
                                             'worst texture',
                                             'worst perimeter',
                                             'worst area',
                                             'worst smoothness',
                                             'worst compactness',
                                             'worst concavity',
                                             'worst concave points',
                                             'worst symmetry',
                                             'worst fractal dimension']


def test_iris_LogisticRegression_ModelDriftExplainer():
    # load iris data
    dataset = datasets.load_iris()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED)
    # build logistic regression model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X=X_train, y=y_train)
    # fit ModelDriftExplainer
    drift_explainer = ModelDriftExplainer(clf, task='classification')
    drift_explainer.fit(X1=X_train, X2=X_test, y1=y_train, y2=y_test)

    # prediction drift "raw"
    prediction_drift_ref = [DriftMetricsNum(mean_difference=0.7557013656904251, wasserstein=0.7613281055658767, ks_test=BaseStatisticalTestResult(statistic=0.12698412698412698, pvalue=0.6467769104301898)),
                            DriftMetricsNum(mean_difference=0.12136523816105171, wasserstein=0.12787543796411713, ks_test=BaseStatisticalTestResult(
                                statistic=0.16825396825396827, pvalue=0.3024954514809168)),
                            DriftMetricsNum(mean_difference=-0.8770666038514756, wasserstein=0.880126585236782, ks_test=BaseStatisticalTestResult(statistic=0.12380952380952381, pvalue=0.6769980003896401))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(prediction_type='raw'),
                                    prediction_drift_ref,
                                    rtol=1e-3, atol=1e-3)

    # prediction drift "proba"
    prediction_drift_proba_ref = [DriftMetricsNum(mean_difference=0.05808348053790602, wasserstein=0.058098961387593524, ks_test=BaseStatisticalTestResult(statistic=0.12698412698412698, pvalue=0.6467769104301898)),
                                  DriftMetricsNum(mean_difference=0.020486971268375176, wasserstein=0.02567105402118564, ks_test=BaseStatisticalTestResult(
                                      statistic=0.07301587301587302, pvalue=0.9917692279981893)),
                                  DriftMetricsNum(mean_difference=-0.07857045180628125, wasserstein=0.07857045186835532, ks_test=BaseStatisticalTestResult(statistic=0.1365079365079365, pvalue=0.5571746191565531))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(prediction_type='proba'),
                                    prediction_drift_proba_ref,
                                    rtol=1e-3, atol=1e-3)

    # prediction drift "class"
    prediction_drift_class_ref = [DriftMetricsCat(wasserstein=0.12380952380952381,
                                                  jensen_shannon=0.09340505952259712,
                                                  chi2_test=Chi2TestResult(statistic=2.113284012922712,
                                                                           pvalue=0.3476211622295921,
                                                                           dof=2,
                                                                           contingency_table=pd.DataFrame([[33.0, 31.0, 41.0], [17.0, 16.0, 12.0]],
                                                                                                          index=['X1', 'X2'], columns=['0', '1', '2'])))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(prediction_type='class'),
                                    prediction_drift_class_ref)

    # target drift
    assert_drift_metrics_equal(drift_explainer.get_target_drift(),
                               DriftMetricsCat(wasserstein=0.09523809523809523,
                                               jensen_shannon=0.07382902143706498,
                                               chi2_test=Chi2TestResult(statistic=1.3333333333333333,
                                                                        pvalue=0.5134171190325922,
                                                                        dof=2,
                                                                        contingency_table=pd.DataFrame([[33.0, 34.0, 38.0], [17.0, 16.0, 12.0]],
                                                                                                       index=['X1', 'X2'], columns=['0', '1', '2']))))

    # model agnostic drift importances "mean"
    assert_allclose(drift_explainer.get_model_agnostic_drift_values(type='mean'),
                    np.array([[0.53218358,  0.03371401, -0.56589759],
                              [-0.53260398, -0.04423636,  0.57684034],
                              [0.69789646,  0.05740799, -0.75530445],
                              [0.75401974,  0.07368188, -0.82770162]]),
                    atol=1e-3)

    # model agnostic drift importances "wasserstein"
    assert_allclose(drift_explainer.get_model_agnostic_drift_values(type='wasserstein'),
                    np.array([[0.56914095, 0.04106184, 0.59482062],
                              [0.54119694, 0.04603877, 0.58388156],
                              [0.69789646, 0.05740799, 0.75530445],
                              [0.75401974, 0.07368188, 0.82770162]]),
                    atol=1e-3)

    assert drift_explainer.cat_feature_indices == []
    assert drift_explainer.class_names == ['0', '1', '2']
    assert drift_explainer.n_features == 4
    assert drift_explainer.task == 'classification'
