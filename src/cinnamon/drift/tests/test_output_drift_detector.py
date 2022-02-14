import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from cinnamon.drift import OutputDriftDetector
from cinnamon.drift.drift_utils import (DriftMetricsNum, DriftMetricsCat,
                                        PerformanceMetricsDrift,
                                        assert_drift_metrics_equal,
                                        assert_drift_metrics_list_equal,
                                        assert_performance_metrics_drift_equal)
from cinnamon.common.stat_utils import (BaseStatisticalTestResult, Chi2TestResult,
                                        RegressionMetrics, ClassificationMetrics)

RANDOM_SEED = 2021


def test_breast_cancer_OutputDriftDetector():
    dataset = datasets.load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
    clf = XGBClassifier(n_estimators=1000,
                        booster="gbtree",
                        objective="binary:logistic",
                        learning_rate=0.05,
                        max_depth=6,
                        use_label_encoder=False,
                        seed=RANDOM_SEED)
    clf.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=0)

    # ---------------------------------
    #  case: prediction_type='proba'
    # ---------------------------------

    output_drift_detector = OutputDriftDetector(task='classification', prediction_type='proba')
    output_drift_detector.fit(clf.predict_proba(X_train), clf.predict_proba(X_test),
                               y_train, y_test)

    # prediction drift
    prediction_drift_ref = [DriftMetricsNum(mean_difference=0.00798452225707491,
                                            wasserstein=0.024082025832758043,
                                            ks_test=BaseStatisticalTestResult(statistic=0.08323782655969908,
                                                                              pvalue=0.3542889176877513))]
    assert_drift_metrics_list_equal(output_drift_detector.get_prediction_drift(),
                                    prediction_drift_ref)

    # target drift
    target_drift_ref = DriftMetricsCat(wasserstein=0.0024097093655411628,
                                       jensen_shannon=0.0017616379091961293,
                                       chi2_test=Chi2TestResult(statistic=0.0,
                                                                pvalue=1.0,
                                                                dof=1,
                                                                contingency_table=pd.DataFrame(
                                                                    [[148.0, 250.0], [64.0, 107.0]],
                                                                    index=['X1', 'X2'], columns=[0, 1])))
    assert_drift_metrics_equal(output_drift_detector.get_target_drift(),
                               target_drift_ref)

    # performance_metrics_drift
    assert_performance_metrics_drift_equal(output_drift_detector.get_performance_metrics_drift(),
                                           PerformanceMetricsDrift(ClassificationMetrics(accuracy=1.0, log_loss=0.016039305599991362),
                                                                   ClassificationMetrics(accuracy=0.9473684210526315, log_loss=0.11116574995208815)))

    # ---------------------------------
    #   case: prediction_type='label'
    # ---------------------------------

    output_drift_detector2 = OutputDriftDetector(task='classification', prediction_type='label')
    output_drift_detector2.fit(clf.predict(X_train), clf.predict(X_test),
                                y_train, y_test)

    # prediction drift
    prediction_drift_ref2 = [DriftMetricsCat(wasserstein=0.015134150283581616,
                                             jensen_shannon=0.011119138338504947,
                                             chi2_test=Chi2TestResult(statistic=0.06175606172739494,
                                                                      pvalue=0.8037416368764607,
                                                                      dof=1,
                                                                      contingency_table=pd.DataFrame(
                                                                          [[148.0, 250.0], [61.0, 110.0]],
                                                                          index=['X1', 'X2'], columns=[0, 1])))]
    assert_drift_metrics_list_equal(output_drift_detector2.get_prediction_drift(),
                                    prediction_drift_ref2)

    # target drift
    assert_drift_metrics_equal(output_drift_detector2.get_target_drift(),
                               target_drift_ref)  # target_drift_ref is the same as in previous case

    # performance_metrics_drift
    assert_performance_metrics_drift_equal(output_drift_detector2.get_performance_metrics_drift(),
                                           PerformanceMetricsDrift(ClassificationMetrics(accuracy=1.0),
                                                                   ClassificationMetrics(accuracy=0.9473684210526315)))

    # ---------------------------------
    #   case: prediction_type='raw'
    # ---------------------------------

    output_drift_detector3 = OutputDriftDetector(task='classification', prediction_type='raw')
    output_drift_detector3.fit(pd.DataFrame(clf.predict(X_train, output_margin=True)),
                                clf.predict(X_test, output_margin=True),
                                y_train, y_test)

    # prediction drift
    prediction_drift_ref3 = [DriftMetricsNum(mean_difference=0.005498574879272855,
                                             wasserstein=0.3764544601013494,
                                             ks_test=BaseStatisticalTestResult(statistic=0.08323782655969908,
                                                                               pvalue=0.3542889176877513))]
    assert_drift_metrics_list_equal(output_drift_detector3.get_prediction_drift(),
                                    prediction_drift_ref3)

    # target drift
    assert_drift_metrics_equal(output_drift_detector3.get_target_drift(),
                               target_drift_ref)  # target_drift_ref is the same as in previous case

    # performance_metrics_drift
    assert_performance_metrics_drift_equal(output_drift_detector3.get_performance_metrics_drift(),
                                           PerformanceMetricsDrift(ClassificationMetrics(accuracy=1.0, log_loss=0.016039305803571925),
                                                                   ClassificationMetrics(accuracy=0.9473684210526315, log_loss=0.11116572790125613)))


def test_iris_OutputDriftDetector():
    dataset = datasets.load_iris()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
    clf = XGBClassifier(n_estimators=1000,
                        booster="gbtree",
                        learning_rate=0.05,
                        max_depth=6,
                        use_label_encoder=False,
                        seed=2021)
    clf.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=0)

    # ---------------------------------
    #  case: prediction_type='proba'
    # ---------------------------------

    output_drift_detector = OutputDriftDetector(task='classification', prediction_type='proba')
    output_drift_detector.fit(clf.predict_proba(X_train), clf.predict_proba(X_test),
                               y_train, y_test)

    # prediction drift
    prediction_drift_ref = [DriftMetricsNum(mean_difference=0.06122570359665486,
                                            wasserstein=0.06138405880136859,
                                            ks_test=BaseStatisticalTestResult(statistic=0.1111111111111111,
                                                                              pvalue=0.793799989988573)),
                            DriftMetricsNum(mean_difference=0.08154049228640303,
                                            wasserstein=0.08205600790975118,
                                            ks_test=BaseStatisticalTestResult(statistic=0.12698412698412698,
                                                                              pvalue=0.6467769104301901)),
                            DriftMetricsNum(mean_difference=-0.1427661934187488,
                                            wasserstein=0.14276781702443725,
                                            ks_test=BaseStatisticalTestResult(statistic=0.19047619047619047,
                                                                              pvalue=0.1805850065949114))]
    assert_drift_metrics_list_equal(output_drift_detector.get_prediction_drift(),
                                    prediction_drift_ref)

    # target drift
    target_drift_ref = DriftMetricsCat(wasserstein=0.09523809523809523,
                                       jensen_shannon=0.07382902143706498,
                                       chi2_test=Chi2TestResult(statistic=1.3333333333333333,
                                                                pvalue=0.5134171190325922,
                                                                dof=2,
                                                                contingency_table=pd.DataFrame(
                                                                    [[33.0, 34.0, 38.0], [17.0, 16.0, 12.0]],
                                                                    index=['X1', 'X2'], columns=[0, 1, 2])))
    assert_drift_metrics_equal(output_drift_detector.get_target_drift(),
                               target_drift_ref)

    # performance_metrics_drift
    assert_performance_metrics_drift_equal(output_drift_detector.get_performance_metrics_drift(),
                                           PerformanceMetricsDrift(ClassificationMetrics(accuracy=1.0, log_loss=0.045063073312242824),
                                                                   ClassificationMetrics(accuracy=0.9333333333333333, log_loss=0.16192325585418277)))

    # ---------------------------------
    #   case: prediction_type='label'
    # ---------------------------------

    output_drift_detector2 = OutputDriftDetector(task='classification', prediction_type='label')
    output_drift_detector2.fit(clf.predict(X_train), clf.predict(X_test),
                                y_train, y_test)

    # prediction drift
    prediction_drift_ref2 = [DriftMetricsCat(wasserstein=0.16190476190476188,
                                             jensen_shannon=0.12861925049715453,
                                             chi2_test=Chi2TestResult(statistic=3.879642904933953,
                                                                      pvalue=0.14372961005414284,
                                                                      dof=2,
                                                                      contingency_table=pd.DataFrame(
                                                                          [[33.0, 34.0, 38.0], [17.0, 19.0, 9.0]],
                                                                          index=['X1', 'X2'], columns=[0, 1, 2])))]
    assert_drift_metrics_list_equal(output_drift_detector2.get_prediction_drift(),
                                    prediction_drift_ref2)

    # target drift
    assert_drift_metrics_equal(output_drift_detector2.get_target_drift(),
                               target_drift_ref)  # target_drift_ref is the same as in previous case

    # performance_metrics_drift
    assert_performance_metrics_drift_equal(output_drift_detector2.get_performance_metrics_drift(),
                                           PerformanceMetricsDrift(ClassificationMetrics(accuracy=1.0),
                                                                   ClassificationMetrics(accuracy=0.9333333333333333)))

    # ---------------------------------
    #   case: prediction_type='raw'
    # ---------------------------------

    output_drift_detector3 = OutputDriftDetector(task='classification', prediction_type='raw')
    output_drift_detector3.fit(pd.DataFrame(clf.predict(X_train, output_margin=True)),
                                clf.predict(X_test, output_margin=True),
                                y_train, y_test)

    # prediction drift
    prediction_drift_ref3 = [DriftMetricsNum(mean_difference=0.31093458145383807,
                                             wasserstein=0.310934581453838,
                                             ks_test=BaseStatisticalTestResult(statistic=0.06349206349206349,
                                                                               pvalue=0.9987212484986797)),
                             DriftMetricsNum(mean_difference=0.3232848411632908,
                                             wasserstein=0.3318073130907522,
                                             ks_test=BaseStatisticalTestResult(statistic=0.12698412698412698,
                                                                               pvalue=0.6467769104301901)),
                             DriftMetricsNum(mean_difference=-0.5564053781212321,
                                             wasserstein=0.5568392310587188,
                                             ks_test=BaseStatisticalTestResult(statistic=0.17142857142857143,
                                                                               pvalue=0.2821678346768163))]
    assert_drift_metrics_list_equal(output_drift_detector3.get_prediction_drift(),
                                    prediction_drift_ref3)

    # target drift
    assert_drift_metrics_equal(output_drift_detector3.get_target_drift(),
                               target_drift_ref)  # target_drift_ref is the same as in previous case

    # performance_metrics_drift
    assert_performance_metrics_drift_equal(output_drift_detector3.get_performance_metrics_drift(),
                                           PerformanceMetricsDrift(ClassificationMetrics(accuracy=1.0, log_loss=0.04506306932086036),
                                                                   ClassificationMetrics(accuracy=0.9333333333333333, log_loss=0.1619232536604007)))


def test_boston_OutputDriftDetector():
    dataset = datasets.load_boston()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
    model = XGBRegressor(n_estimators=1000,
                         booster="gbtree",
                         objective="reg:squarederror",
                         learning_rate=0.05,
                         max_depth=6,
                         seed=RANDOM_SEED,
                         use_label_encoder=False)
    model.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=0)

    output_drift_detector = OutputDriftDetector(task='regression')
    output_drift_detector.fit(model.predict(X_train), model.predict(X_test), y_train, y_test)

    # prediction drift
    prediction_drift_ref = [DriftMetricsNum(mean_difference=-0.7889487954289152,
                                            wasserstein=1.0808420273082935,
                                            ks_test=BaseStatisticalTestResult(statistic=0.052743086529884034,
                                                                              pvalue=0.9096081584010306))]
    assert_drift_metrics_list_equal(output_drift_detector.get_prediction_drift(),
                                    prediction_drift_ref)

    # target drift
    target_drift_ref = DriftMetricsNum(mean_difference=-0.609240261671129,
                                       wasserstein=1.3178114778471604,
                                       ks_test=BaseStatisticalTestResult(statistic=0.07857567647933393,
                                                                         pvalue=0.4968030078636394))
    assert_drift_metrics_equal(output_drift_detector.get_target_drift(),
                               target_drift_ref)

    # performance_metrics_drift
    assert_performance_metrics_drift_equal(output_drift_detector.get_performance_metrics_drift(),
                                           PerformanceMetricsDrift(RegressionMetrics(mse=0.3643813701486243,
                                                                                     explained_variance=0.9960752192224699),
                                                                   RegressionMetrics(mse=12.419719495108291,
                                                                                     explained_variance=0.8095694395593922)))
