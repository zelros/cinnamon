import pandas as pd
import numpy as np
from numpy.testing import assert_allclose

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_openml
from sklearn.compose import make_column_selector as selector, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from cinnamon.drift import ModelDriftExplainer
from cinnamon.drift.drift_utils import (DriftMetricsNum,
                                        PerformanceMetricsDrift,
                                        assert_drift_metrics_list_equal,
                                        assert_drift_metrics_equal,
                                        assert_performance_metrics_drift_equal)
from cinnamon.common.stat_utils import (
    BaseStatisticalTestResult, RegressionMetrics)
from ...common.constants import NUMPY_atol

RANDOM_SEED = 2021


def test_AmesHousing_LinearRegression_ModelDriftExplainer():
    # load ames housing data
    try:
        ames_housing = fetch_openml(name="house_prices", as_frame=True)
    except:
        # the following code solves the error that may occur because of SSL certificates
        # https://github.com/scikit-learn/scikit-learn/issues/10201#issuecomment-365734582
        import os
        import ssl
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
                getattr(ssl, '_create_unverified_context', None)):
            ssl._create_default_https_context = ssl._create_unverified_context

        ames_housing = fetch_openml(name="house_prices", as_frame=True)

    ames_housing_df = pd.DataFrame(
        data=ames_housing.data, columns=ames_housing.feature_names)

    # preprocessing: drop columns with missing values
    dropped_columns = ames_housing_df.columns[ames_housing_df.isnull().sum(
        axis=0) > 0]
    ames_housing_df.drop(dropped_columns, axis=1, inplace=True)

    # build linear model pipeline
    # use one hot encoding to preprocess the categorical columms
    categorical_columns_selector = selector(dtype_include=object)
    categorical_columns = categorical_columns_selector(ames_housing_df)
    preprocessor = ColumnTransformer([
        ('one-hot-encoder', OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ])
    pipe = make_pipeline(preprocessor, LinearRegression())
    # train the pipe
    X_train, X_test, y_train, y_test = train_test_split(
        ames_housing_df, ames_housing.target, test_size=0.3, random_state=RANDOM_SEED)
    pipe.fit(X=X_train, y=y_train)

    # fit ModelDriftExplainer
    drift_explainer = ModelDriftExplainer(pipe, task='regression')
    cat_feature_indices = [ames_housing_df.columns.to_list().index(
        name) for name in categorical_columns]
    drift_explainer.fit(X_train, X_test, y_train, y_test,
                        cat_feature_indices=cat_feature_indices)

    # prediction drift
    prediction_drift_ref = [DriftMetricsNum(mean_difference=-4719.931485324632,
                                            wasserstein=5368.068091932769,
                                            ks_test=BaseStatisticalTestResult(statistic=0.046966731898238745, pvalue=0.4943709245152732))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(),
                                    prediction_drift_ref)

    # target drift
    assert_drift_metrics_equal(drift_explainer.get_target_drift(),
                               DriftMetricsNum(mean_difference=-5589.04044357469, wasserstein=6036.554468362687, ks_test=BaseStatisticalTestResult(statistic=0.04044357469015003, pvalue=0.6829578717438207)))

    # performance_metrics_drift
    assert_performance_metrics_drift_equal(drift_explainer.get_performance_metrics_drift(),
                                           PerformanceMetricsDrift(RegressionMetrics(mse=1297832435.0235813,
                                                                                     explained_variance=0.7990458696049108),
                                                                   RegressionMetrics(mse=1539552291.4610596,
                                                                                     explained_variance=0.7405622136312852)))

    # model agnostic drift importances "mean"
    assert_allclose(drift_explainer.get_model_agnostic_drift_values(type='mean'),
                    np.array([[22.09852279],
                              [-2742.93098179],
                              [-1272.03892043],
                              [1676.60830741],
                              [139.27557737],
                              [-634.86117881],
                              [416.50419545],
                              [19.8811366],
                              [-239.09984374],
                              [109.73681169],
                              [-2884.27844107],
                              [-1101.22823766],
                              [-665.71429727],
                              [-875.2803564],
                              [-1481.31342991],
                              [-4274.15295932],
                              [535.13877096],
                              [-307.61412696],
                              [330.09848755],
                              [-1106.32960826],
                              [925.79167026],
                              [-702.02373115],
                              [-875.19509628],
                              [1712.35873927],
                              [399.74873242],
                              [-1468.23359607],
                              [-3292.11558712],
                              [250.45032064],
                              [2359.93160402],
                              [-77.03018287],
                              [-152.78242262],
                              [-11.943687],
                              [-1155.44720966],
                              [557.79079527],
                              [-1333.76459887],
                              [-208.89529464],
                              [-2096.35745397],
                              [-128.73986359],
                              [33.42124815],
                              [-1138.82393542],
                              [-3159.05562056],
                              [-1513.58831508],
                              [-708.64973275],
                              [-2028.90107999],
                              [35.00279819],
                              [-446.50381884],
                              [-897.04023941],
                              [91.43054676],
                              [-1423.19514136],
                              [-902.33576606],
                              [-1984.42785558],
                              [-3757.42205693],
                              [-1368.93184387],
                              [154.36933817],
                              [164.60699722],
                              [-121.26534749],
                              [-80.99440624],
                              [240.82061511],
                              [191.31579472],
                              [-613.81116551],
                              [-791.99099392]]),
                    atol=NUMPY_atol)

    # model agnostic drift importances "mean"
    assert_allclose(drift_explainer.get_model_agnostic_drift_values(type='wasserstein'),
                    np.array([[132.7379132],
                              [2742.93098179],
                              [1273.53420285],
                              [1676.60830741],
                              [140.62682802],
                              [649.98619169],
                              [434.35465174],
                              [25.9197543],
                              [597.47441859],
                              [186.25823344],
                              [2884.27844108],
                              [1141.79674059],
                              [678.14704706],
                              [891.9521385],
                              [1481.31342991],
                              [4274.15295932],
                              [802.93448323],
                              [524.39921364],
                              [873.09605712],
                              [1149.59100332],
                              [945.15573531],
                              [965.18980404],
                              [1000.58838323],
                              [1712.35873927],
                              [444.65560041],
                              [1468.28554851],
                              [3292.11558712],
                              [515.90950568],
                              [2427.06709658],
                              [1180.59310563],
                              [236.67699021],
                              [395.33201824],
                              [1155.51733869],
                              [1373.27521973],
                              [1885.06174847],
                              [310.50833521],
                              [2558.76522263],
                              [339.57889779],
                              [124.03291044],
                              [1255.44267436],
                              [3159.05562056],
                              [1513.58831508],
                              [711.74894287],
                              [2067.72247249],
                              [1329.51719256],
                              [649.61491411],
                              [1171.66018451],
                              [1006.82318531],
                              [1808.74164757],
                              [903.20221618],
                              [1984.73941976],
                              [3758.85510456],
                              [1609.86450235],
                              [357.21692829],
                              [622.22580499],
                              [255.87472293],
                              [262.93929184],
                              [674.65616464],
                              [407.72131898],
                              [638.44254154],
                              [799.50356507]]),
                    atol=NUMPY_atol)

    assert drift_explainer.cat_feature_indices == [2,
                                                   4,
                                                   5,
                                                   6,
                                                   7,
                                                   8,
                                                   9,
                                                   10,
                                                   11,
                                                   12,
                                                   13,
                                                   14,
                                                   19,
                                                   20,
                                                   21,
                                                   22,
                                                   23,
                                                   24,
                                                   25,
                                                   30,
                                                   31,
                                                   32,
                                                   43,
                                                   45,
                                                   49,
                                                   59,
                                                   60]
