import pandas as pd, numpy as np
from numpy.testing import assert_allclose
from sklearn import datasets
from sklearn.model_selection import train_test_split
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


def test_boston_catboost_ModelDriftExplainer():
    boston = datasets.load_boston()
    boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(boston_df, boston.target, test_size=0.3,
                                                        random_state=RANDOM_SEED)
    model = CatBoostRegressor(loss_function='RMSE',
                              learning_rate=0.1,
                              iterations=1000,
                              max_depth=6,
                              early_stopping_rounds=20,
                              random_seed=RANDOM_SEED,
                              verbose=False)
    model.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)])
    drift_explainer = ModelDriftExplainer(model)
    drift_explainer.fit(X_train, X_test, y_train, y_test)

    # prediction drift
    prediction_drift_ref = [DriftMetricsNum(mean_difference=-0.8779339801393569, wasserstein=1.250335807497859, ks_test=BaseStatisticalTestResult(statistic=0.05764942016057092, pvalue=0.8464257838033299))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(),
                                    prediction_drift_ref)

    # target drift
    assert_drift_metrics_equal(drift_explainer.get_target_drift(),
                               DriftMetricsNum(mean_difference=-0.609240261671129, wasserstein=1.3178114778471604, ks_test=BaseStatisticalTestResult(statistic=0.07857567647933393, pvalue=0.4968030078636394)))

    # performance_metrics_drift
    assert_performance_metrics_drift_equal(drift_explainer.get_performance_metrics_drift(),
                                           PerformanceMetricsDrift(RegressionMetrics(mse=1.9584429193171917,
                                                                                     explained_variance=0.9788618728948271),
                                                                   RegressionMetrics(mse=9.57426222359859,
                                                                                     explained_variance=0.8537685044683866)))

    # tree_based_drift_values "node_size"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='node_size'),
                    np.array([[1.90483198],
                              [0.42289995],
                              [0.84419323],
                              [0.40268871],
                              [2.2042409 ],
                              [2.75792564],
                              [1.13996793],
                              [2.4437588 ],
                              [1.46726763],
                              [1.05893021],
                              [1.13820808],
                              [1.9729567 ],
                              [4.08308837]]),
                    atol=NUMPY_atol)

    # tree_based_drift_values "mean"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='mean'),
                    np.array([[ 0.06869412],
                              [-0.00890033],
                              [ 0.00371266],
                              [ 0.04273852],
                              [ 0.09479128],
                              [-0.34271389],
                              [-0.04711113],
                              [-0.03455269],
                              [-0.09711201],
                              [-0.0245336 ],
                              [-0.07489948],
                              [-0.09540242],
                              [-0.362645  ]]),
                    atol=NUMPY_atol)

    # tree_based_drift_values "mean_norm"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='mean_norm'),
                    np.array([[ 0.0028605 ],
                              [-0.01031604],
                              [-0.02142061],
                              [ 0.03696658],
                              [ 0.03681141],
                              [-0.3557283 ],
                              [-0.04529067],
                              [-0.1080007 ],
                              [-0.09353176],
                              [-0.02250316],
                              [-0.07035532],
                              [-0.029536  ],
                              [-0.43489742]]),
                    atol=NUMPY_atol)

    # feature_drift_LSTAT
    assert_drift_metrics_equal(drift_explainer.get_feature_drift('LSTAT'),
                               DriftMetricsNum(mean_difference=0.7378638864109419, wasserstein=0.8023078352661315, ks_test=BaseStatisticalTestResult(statistic=0.08887154326494201, pvalue=0.3452770147763923)))

    # feature_drift_feature_0
    assert_drift_metrics_equal(drift_explainer.get_feature_drift(0),
                               DriftMetricsNum(mean_difference=-1.1253475613291695, wasserstein=1.1305975918079103, ks_test=BaseStatisticalTestResult(statistic=0.0618123699078204, pvalue=0.7813257636577198)))

    # all feature drifts
    feature_drifts_ref = [DriftMetricsNum(mean_difference=-1.1253475613291695, wasserstein=1.1305975918079103, ks_test=BaseStatisticalTestResult(statistic=0.0618123699078204, pvalue=0.7813257636577198)),
                          DriftMetricsNum(mean_difference=-0.7548691644365135, wasserstein=0.8747398156407966, ks_test=BaseStatisticalTestResult(statistic=0.02951234017246506, pvalue=0.9999448615410187)),
                          DriftMetricsNum(mean_difference=-0.17914585191793364, wasserstein=0.7212444246208749, ks_test=BaseStatisticalTestResult(statistic=0.07381801962533452, pvalue=0.5764298225321844)),
                          DriftMetricsNum(mean_difference=0.02337942313410645, wasserstein=0.02337942313410646, ks_test=BaseStatisticalTestResult(statistic=0.02337942313410645, pvalue=0.9999998980177341)),
                          DriftMetricsNum(mean_difference=-0.010584444692238959, wasserstein=0.01748104742789176, ks_test=BaseStatisticalTestResult(statistic=0.07296312815938151, pvalue=0.5907168134992118)),
                          DriftMetricsNum(mean_difference=-0.015463871543265562, wasserstein=0.07063187630092177, ks_test=BaseStatisticalTestResult(statistic=0.056645851917930416, pvalue=0.86067051746252)),
                          DriftMetricsNum(mean_difference=-0.5575044603033064, wasserstein=1.22835637823372, ks_test=BaseStatisticalTestResult(statistic=0.04114629794826048, pvalue=0.9894125780752614)),
                          DriftMetricsNum(mean_difference=0.12717342030924828, wasserstein=0.17319668822479928, ks_test=BaseStatisticalTestResult(statistic=0.07244275944097532, pvalue=0.5998930415655818)),
                          DriftMetricsNum(mean_difference=-0.28690900981266765, wasserstein=0.3868941421349984, ks_test=BaseStatisticalTestResult(statistic=0.03631430270591734, pvalue=0.9978241882813342)),
                          DriftMetricsNum(mean_difference=-13.692387749033628, wasserstein=14.388492417484393, ks_test=BaseStatisticalTestResult(statistic=0.08039696699375558, pvalue=0.4679944796162433)),
                          DriftMetricsNum(mean_difference=0.20367603330359785, wasserstein=0.205839280404401, ks_test=BaseStatisticalTestResult(statistic=0.06322479928635147, pvalue=0.7594689122671989)),
                          DriftMetricsNum(mean_difference=6.062043190603617, wasserstein=6.4615822925958994, ks_test=BaseStatisticalTestResult(statistic=0.06341064525721082, pvalue=0.7559278097560319)),
                          DriftMetricsNum(mean_difference=0.7378638864109419, wasserstein=0.8023078352661315, ks_test=BaseStatisticalTestResult(statistic=0.08887154326494201, pvalue=0.3452770147763923))]
    assert_drift_metrics_list_equal(drift_explainer.get_feature_drifts(),
                                    feature_drifts_ref)

    # tree_based_correction_weights with default params
    assert_allclose(drift_explainer.get_tree_based_correction_weights(),
                    np.array([1.26204467, 1.12758195, 1.1098125 , 1.13271794, 1.13462246,
                              1.1165066 , 1.13961283, 1.11612841, 0.6649825 , 0.9101491 ,
                              1.11189125, 1.03815182, 1.03340836, 1.10880066, 0.75487182,
                              0.62898398, 1.07190854, 0.97157506, 0.72438018, 0.73757673,
                              1.21736199, 0.93459267, 1.12398518, 1.04348364, 0.96185557,
                              1.1577552 , 1.01165243, 1.11628833, 0.91056533, 0.83375788,
                              0.70681885, 1.01313382, 1.31589968, 1.01646162, 1.0793867 ,
                              1.05104272, 1.08126388, 1.06460182, 1.06142385, 1.08647292,
                              1.0602133 , 1.08808216, 1.06097316, 1.0085106 , 0.71690766,
                              1.0964748 , 1.12949296, 1.13137799, 1.12525403, 0.79226995,
                              1.1735033 , 1.0767666 , 0.7389624 , 0.92409138, 1.02928402,
                              0.87390194, 1.04918248, 0.98114919, 0.90461318, 0.99690141,
                              1.04944021, 0.84043304, 0.91081296, 0.55272801, 1.06356075,
                              1.04406892, 1.14332477, 0.95402909, 1.07448815, 1.06314665,
                              1.15305845, 1.08667437, 1.08295325, 0.61808778, 0.93841517,
                              1.05701348, 1.12788455, 0.89833351, 1.03701816, 0.76605064,
                              0.79993235, 1.01853021, 1.00812235, 0.68440514, 1.08131822,
                              1.15944491, 1.10028033, 1.02133312, 0.77126167, 1.1169331 ,
                              0.87226883, 1.0274817 , 1.01806625, 0.8738874 , 0.82350401,
                              1.05955957, 0.90911545, 0.40651477, 1.04803514, 1.12623313,
                              1.12227116, 1.14571827, 1.04230305, 1.12393889, 1.16377785,
                              0.82612925, 1.01915541, 1.01503278, 0.8620892 , 0.98212787,
                              1.03699855, 1.15657293, 1.06550115, 1.0064774 , 1.11731286,
                              0.85131301, 1.20523173, 0.99942746, 0.80044317, 0.71548786,
                              1.10341947, 0.93963463, 0.61569436, 1.00081533, 1.08776728,
                              1.07295692, 1.10010529, 0.87705107, 0.71708581, 0.92747423,
                              0.95394448, 1.01897125, 1.07292986, 1.09759025, 1.05769623,
                              1.05323663, 1.14615865, 0.75403556, 0.83882016, 0.93634525,
                              1.08743475, 1.18833951, 0.99177265, 1.05426211, 0.78516348,
                              0.48991769, 0.73260998, 1.0319473 , 0.98482995, 0.94403372,
                              1.37041794, 1.11271977, 1.07681627, 1.13313276, 0.37633866,
                              0.6236818 , 1.10345547, 1.05265211, 1.1262599 , 1.08363078,
                              0.94800955, 1.10693143, 0.98511127, 1.06251833, 1.27042363,
                              0.71259725, 0.92953185, 1.22249787, 1.13422245, 1.08934502,
                              1.01907369, 1.09034165, 0.99168541, 1.09420003, 1.07536856,
                              0.8122953 , 1.12414347, 1.00207014, 1.05484035, 0.90979186,
                              0.79486343, 0.88513653, 1.13279839, 1.14051842, 0.83545873,
                              1.26441103, 1.05784774, 1.00351022, 0.97224466, 1.08296292,
                              1.12561801, 1.15469868, 1.04479062, 0.98524393, 1.14416384,
                              1.16673851, 1.05295793, 1.0901144 , 0.98444583, 1.27904676,
                              0.97641288, 0.93157474, 0.92839532, 0.8393623 , 1.00766852,
                              1.16061036, 0.60164609, 1.19944905, 0.82551414, 1.10334377,
                              1.0347387 , 1.01459291, 0.60733288, 1.07325684, 0.94225207,
                              0.81292041, 0.79177656, 0.97445209, 1.03076143, 1.11147617,
                              0.84454846, 0.79125186, 1.05158967, 0.94814929, 0.83491393,
                              1.03337443, 1.08380137, 1.08361233, 1.06407962, 1.05943645,
                              1.11413293, 0.93745831, 1.01865618, 1.0756428 , 1.09025293,
                              1.34224054, 1.08469619, 0.99515584, 0.99791534, 1.14609149,
                              1.10220546, 1.10655145, 1.02036823, 1.10529888, 1.27661895,
                              1.20303298, 1.11570198, 1.04415163, 0.75277445, 1.03027215,
                              0.89199167, 1.10340212, 1.04283617, 0.8981111 , 0.42991246,
                              1.09663151, 1.256944  , 1.24974654, 1.10685389, 0.79354265,
                              0.91414439, 1.13016238, 0.89852017, 1.06255957, 0.92109072,
                              0.75752724, 0.99058815, 0.96723553, 1.11404277, 0.90148599,
                              0.83184323, 0.97127245, 0.81092146, 0.64699283, 0.70473045,
                              1.11558755, 0.8553589 , 1.02399408, 1.10955695, 0.86333844,
                              1.04051355, 1.16375158, 1.14611548, 1.19964363, 1.21960475,
                              1.05222397, 1.01633969, 1.17325167, 1.14014116, 0.92470836,
                              1.01147851, 1.20072323, 1.1313567 , 0.82869624, 1.05976579,
                              1.23110071, 1.0830754 , 0.79357543, 1.08449987, 0.96710123,
                              0.82443848, 0.98763578, 1.03064906, 1.07669648, 1.08406061,
                              1.07438381, 1.12997196, 0.99596552, 0.90128289, 1.12868713,
                              1.17248793, 0.97608255, 1.11212892, 0.9704317 , 1.06249169,
                              0.98339119, 1.10227236, 1.03327927, 0.92720463, 1.08533577,
                              0.98078947, 1.10147139, 0.67812663, 1.11703186, 1.10053492,
                              0.79891288, 0.89857003, 0.54973991, 1.03118822, 0.76857038,
                              1.14446561, 0.90206737, 0.89310273, 1.02372394, 1.09747072,
                              1.22432677, 1.12042835, 0.82996937, 0.94792183, 1.07949222,
                              0.71751433, 0.93075069, 1.15650086, 0.92698566, 1.13077359,
                              1.08510617, 1.0996725 , 1.03192748, 0.81301031, 1.12839233,
                              1.02642602, 0.98966379, 0.84655256, 1.07963336]),
                    atol=NUMPY_atol)

    # sample_weights1
    assert_allclose(drift_explainer.sample_weights1[:5],
                    np.array([1., 1., 1., 1., 1.]),
                    atol=NUMPY_atol)

    # sample_weights2
    assert_allclose(drift_explainer.sample_weights2[:5],
                    np.array([1., 1., 1., 1., 1.]),
                    atol=NUMPY_atol)

    assert drift_explainer.cat_feature_indices == []
    assert drift_explainer.class_names == []
    assert drift_explainer.feature_names == ['CRIM',
                                             'ZN',
                                             'INDUS',
                                             'CHAS',
                                             'NOX',
                                             'RM',
                                             'AGE',
                                             'DIS',
                                             'RAD',
                                             'TAX',
                                             'PTRATIO',
                                             'B',
                                             'LSTAT']
    assert drift_explainer.iteration_range == (0, 123)
    assert drift_explainer.n_features == 13
    assert drift_explainer.task == 'regression'


def test_breast_cancer_catboost_ModelDriftExplainer():
    dataset = datasets.load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
    clf = CatBoostClassifier(loss_function='Logloss',
                             learning_rate=0.1,
                             iterations=1000,
                             max_depth=6,
                             early_stopping_rounds=20,
                             random_seed=RANDOM_SEED,
                             verbose=False)
    clf.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)])
    drift_explainer = ModelDriftExplainer(clf)
    drift_explainer.fit(X1=X_train, X2=X_test, y1=y_train, y2=y_test)

    # prediction drift "raw"
    prediction_drift_ref = [DriftMetricsNum(mean_difference=0.04634869138730369, wasserstein=0.5538123284880192, ks_test=BaseStatisticalTestResult(statistic=0.0658702871080549, pvalue=0.6459307033330981))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(),
                                    prediction_drift_ref)

    # prediction drift "proba"
    prediction_drift_proba_ref = [DriftMetricsNum(mean_difference=0.016349784083660057, wasserstein=0.0339516727427027, ks_test=BaseStatisticalTestResult(statistic=0.0658702871080549, pvalue=0.6459307033330981))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(prediction_type='proba'),
                                    prediction_drift_proba_ref)

    # target drift
    assert_drift_metrics_equal(drift_explainer.get_target_drift(),
                               DriftMetricsCat(wasserstein=0.0024097093655411628,
                                               jensen_shannon=0.0017616379091961293,
                                               chi2_test=Chi2TestResult(statistic=0.0,
                                                                        pvalue=1.0,
                                                                        dof=1,
                                                                        contingency_table=pd.DataFrame([[148.0, 250.0], [64.0, 107.0]],
                                                                                                       index=['X1', 'X2'], columns=[0, 1]))))

    # performance_metrics_drift
    assert_performance_metrics_drift_equal(drift_explainer.get_performance_metrics_drift(),
                                           PerformanceMetricsDrift(ClassificationMetrics(accuracy=1.0, log_loss=0.00748801449692523),
                                                                   ClassificationMetrics(accuracy=0.9590643274853801, log_loss=0.10225262245942275)))


    # tree_based_drift_values "node_size"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='node_size'),
                    np.array([[0.32509379],
                              [1.32677547],
                              [0.80774132],
                              [0.66001433],
                              [0.32493723],
                              [0.45099504],
                              [0.61364896],
                              [0.90952567],
                              [0.34701315],
                              [0.44619327],
                              [0.50151929],
                              [0.76307014],
                              [0.47727041],
                              [0.49002657],
                              [0.51454348],
                              [0.97118271],
                              [0.41764427],
                              [0.49172829],
                              [0.36100245],
                              [0.41368567],
                              [0.83342952],
                              [0.93508863],
                              [0.80483778],
                              [0.95622554],
                              [1.29879061],
                              [0.41801119],
                              [1.00313047],
                              [1.51256486],
                              [0.74948057],
                              [0.48788657]]),
                    atol=NUMPY_atol)

    # tree_based_drift_values "mean"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='mean'),
                    np.array([[-0.00512973],
                              [-0.07137614],
                              [-0.03603572],
                              [-0.00214852],
                              [ 0.00424277],
                              [ 0.0151429 ],
                              [ 0.06059671],
                              [ 0.09650109],
                              [ 0.00792461],
                              [ 0.0054325 ],
                              [-0.02852552],
                              [ 0.00407939],
                              [-0.03285494],
                              [ 0.00963248],
                              [-0.01979541],
                              [ 0.01209277],
                              [-0.00048218],
                              [ 0.02427688],
                              [-0.00064145],
                              [-0.03105579],
                              [-0.04912936],
                              [-0.06080663],
                              [-0.03693812],
                              [-0.03793038],
                              [ 0.07092418],
                              [ 0.02045349],
                              [ 0.04684453],
                              [ 0.1217826 ],
                              [-0.01263006],
                              [-0.02809828]]),
                    atol=NUMPY_atol)

    # tree_based_drift_values "mean_norm"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='mean_norm'),
                    np.array([[-0.00105982],
                              [-0.07325502],
                              [-0.03451459],
                              [-0.02184468],
                              [ 0.00142696],
                              [ 0.00066276],
                              [ 0.06964488],
                              [ 0.05645905],
                              [-0.00341184],
                              [-0.00472395],
                              [-0.02023157],
                              [-0.01013813],
                              [-0.03178182],
                              [-0.00809401],
                              [-0.01765605],
                              [-0.01061387],
                              [ 0.00070906],
                              [ 0.01189473],
                              [ 0.00576327],
                              [-0.01932325],
                              [-0.04673837],
                              [-0.03106766],
                              [-0.04590803],
                              [-0.05081389],
                              [ 0.04648351],
                              [-0.00369685],
                              [ 0.02293893],
                              [ 0.01151517],
                              [-0.0271206 ],
                              [-0.00963751]]),
                    atol=NUMPY_atol)

    # feature_drift mean perimeter
    assert_drift_metrics_equal(drift_explainer.get_feature_drift('mean perimeter'),
                               DriftMetricsNum(mean_difference=-0.5394598724617197,
                                               wasserstein=3.3009656469481903,
                                               ks_test=BaseStatisticalTestResult(statistic=0.07401040289165124,
                                                                                 pvalue=0.4998149146505402)))

    # feature_drift_feature_4
    assert_drift_metrics_equal(drift_explainer.get_feature_drift(4),
                               DriftMetricsNum(mean_difference=-0.0011612596608774894,
                                               wasserstein=0.0016581176349584157,
                                               ks_test=BaseStatisticalTestResult(statistic=0.10195715419201269,
                                                                                 pvalue=0.1528888570418342)))

    # all feature drifts
    feature_drifts_ref = [DriftMetricsNum(mean_difference=-0.06832955714243738, wasserstein=0.5165887184460309, ks_test=BaseStatisticalTestResult(statistic=0.07067501248934732, pvalue=0.5585303796186362)),
                          DriftMetricsNum(mean_difference=0.014966205295483093, wasserstein=0.48581195450938897, ks_test=BaseStatisticalTestResult(statistic=0.07262922801140204, pvalue=0.5235102002710166)),
                          DriftMetricsNum(mean_difference=-0.5394598724617197, wasserstein=3.3009656469481903, ks_test=BaseStatisticalTestResult(statistic=0.07401040289165124, pvalue=0.4998149146505402)),
                          DriftMetricsNum(mean_difference=-17.377639660289788, wasserstein=51.29123100884536, ks_test=BaseStatisticalTestResult(statistic=0.06404831173410915, pvalue=0.6798041178199972)),
                          DriftMetricsNum(mean_difference=-0.0011612596608774894, wasserstein=0.0016581176349584157, ks_test=BaseStatisticalTestResult(statistic=0.10195715419201269, pvalue=0.1528888570418342)),
                          DriftMetricsNum(mean_difference=-0.003067542684181135, wasserstein=0.006741856651679452, ks_test=BaseStatisticalTestResult(statistic=0.06678127479502777, pvalue=0.6292065348385405)),
                          DriftMetricsNum(mean_difference=-0.006204282970407593, wasserstein=0.006991872085574072, ks_test=BaseStatisticalTestResult(statistic=0.05695142378559464, pvalue=0.8061165226745349)),
                          DriftMetricsNum(mean_difference=-0.002757498369038165, wasserstein=0.0030981202797613783, ks_test=BaseStatisticalTestResult(statistic=0.07233536101560434, pvalue=0.5289539515277676)),
                          DriftMetricsNum(mean_difference=-0.0015799582708866666, wasserstein=0.0029724940491933365, ks_test=BaseStatisticalTestResult(statistic=0.07430426988744894, pvalue=0.4945853439694361)),
                          DriftMetricsNum(mean_difference=-0.0005908483940168588, wasserstein=0.0007386144171148141, ks_test=BaseStatisticalTestResult(statistic=0.08001998295571425, pvalue=0.4016092513985777)),
                          DriftMetricsNum(mean_difference=0.0037102162861089028, wasserstein=0.024305223485850316, ks_test=BaseStatisticalTestResult(statistic=0.07703723294836756, pvalue=0.4491840353305204)),
                          DriftMetricsNum(mean_difference=0.02886939669105759, wasserstein=0.05198642040612427, ks_test=BaseStatisticalTestResult(statistic=0.0476211466690176, pvalue=0.9335508976473271)),
                          DriftMetricsNum(mean_difference=0.07566076581739134, wasserstein=0.21045070968879498, ks_test=BaseStatisticalTestResult(statistic=0.07721355314584619, pvalue=0.44639850985093277)),
                          DriftMetricsNum(mean_difference=-1.0084665432425268, wasserstein=3.717517793646603, ks_test=BaseStatisticalTestResult(statistic=0.07694907284962826, pvalue=0.4505043319421054)),
                          DriftMetricsNum(mean_difference=-0.0003356947309647645, wasserstein=0.0005335116518263824, ks_test=BaseStatisticalTestResult(statistic=0.0786828881248347, pvalue=0.4227052366505618)),
                          DriftMetricsNum(mean_difference=-0.0007019634870257772, wasserstein=0.0019131883246642555, ks_test=BaseStatisticalTestResult(statistic=0.08865967263216668, pvalue=0.28293492039886137)),
                          DriftMetricsNum(mean_difference=-0.0013409832539892468, wasserstein=0.003182003266331656, ks_test=BaseStatisticalTestResult(statistic=0.08345822680654735, pvalue=0.3510440105305035)),
                          DriftMetricsNum(mean_difference=2.3271988597960494e-05, wasserstein=0.0006830131799347609, ks_test=BaseStatisticalTestResult(statistic=0.0452555173528461, pvalue=0.9552787666453139)),
                          DriftMetricsNum(mean_difference=0.0011282174468835414, wasserstein=0.0012097365629316194, ks_test=BaseStatisticalTestResult(statistic=0.07890328837168298, pvalue=0.41933341366069177)),
                          DriftMetricsNum(mean_difference=-0.0002275110538070466, wasserstein=0.000301250179258867, ks_test=BaseStatisticalTestResult(statistic=0.05733345088013165, pvalue=0.7993870994743999)),
                          DriftMetricsNum(mean_difference=-0.06736449792823507, wasserstein=0.5360777131270384, ks_test=BaseStatisticalTestResult(statistic=0.05892033265743924, pvalue=0.772495994489099)),
                          DriftMetricsNum(mean_difference=-0.1677551500191079, wasserstein=0.4373399159540389, ks_test=BaseStatisticalTestResult(statistic=0.05468864791795233, pvalue=0.8423784026321328)),
                          DriftMetricsNum(mean_difference=-0.2752897528578728, wasserstein=3.3780036145640473, ks_test=BaseStatisticalTestResult(statistic=0.0704986922918687, pvalue=0.5621042713248987)),
                          DriftMetricsNum(mean_difference=-20.36132563401793, wasserstein=61.92498604131765, ks_test=BaseStatisticalTestResult(statistic=0.057495077727820386, pvalue=0.7963855471533337)),
                          DriftMetricsNum(mean_difference=-0.0031193050045549287, wasserstein=0.0032626057186517362, ks_test=BaseStatisticalTestResult(statistic=0.09193628963531106, pvalue=0.24521731583934214)),
                          DriftMetricsNum(mean_difference=-0.006255693526110079, wasserstein=0.013996743806753077, ks_test=BaseStatisticalTestResult(statistic=0.09212730318257957, pvalue=0.24304491278560958)),
                          DriftMetricsNum(mean_difference=-0.014588629316171553, wasserstein=0.01665632422345645, ks_test=BaseStatisticalTestResult(statistic=0.08436921449352024, pvalue=0.3384205412836392)),
                          DriftMetricsNum(mean_difference=-0.0035156151077022635, wasserstein=0.005018610787857414, ks_test=BaseStatisticalTestResult(statistic=0.04593141144318082, pvalue=0.9493671446153595)),
                          DriftMetricsNum(mean_difference=0.001843304240500776, wasserstein=0.007164130594492942, ks_test=BaseStatisticalTestResult(statistic=0.07614093861118458, pvalue=0.4637484909517612)),
                          DriftMetricsNum(mean_difference=-0.0014582808780746054, wasserstein=0.0027211234535249344, ks_test=BaseStatisticalTestResult(statistic=0.07605277851244527, pvalue=0.46509038807816905))]

    assert_drift_metrics_list_equal(drift_explainer.get_feature_drifts(),
                                    feature_drifts_ref)

    # tree_based_correction_weights with default params
    assert_allclose(drift_explainer.get_tree_based_correction_weights(),
                    np.array([0.92165749, 1.0180526 , 1.01528637, 1.15786976, 1.15251736,
                              1.10880001, 1.08356881, 0.68413624, 0.9424286 , 0.98766633,
                              1.41433958, 1.08706384, 0.97873393, 1.10629262, 1.01031163,
                              0.95405252, 1.0493201 , 1.1201864 , 0.99148762, 0.98701252,
                              1.08090489, 1.05935291, 0.92962193, 0.99033665, 1.12470821,
                              0.97368892, 1.03117403, 1.11728288, 1.01424376, 1.0172446 ,
                              1.15340733, 1.04214744, 1.00748081, 0.99182202, 0.99066358,
                              0.9775729 , 0.88190388, 0.99897512, 0.85684419, 1.11343334,
                              1.00427673, 1.0137275 , 1.02612415, 1.02836285, 0.96133991,
                              0.90516637, 1.08759653, 0.98257095, 1.02172571, 0.91802906,
                              0.93808599, 0.99696338, 1.07369889, 1.05587756, 0.83185497,
                              0.97812855, 1.04413349, 1.01891141, 0.90346127, 0.99362902,
                              0.9684525 , 0.90370031, 1.10388789, 0.92171006, 0.89300965,
                              0.98234068, 0.96709105, 1.07491421, 1.0715786 , 0.91088615,
                              0.98381211, 1.05858935, 0.97279471, 1.05012173, 0.96044355,
                              0.99105465, 0.91613061, 1.01055213, 1.0386696 , 0.94040953,
                              1.13212958, 0.95137864, 0.97603354, 1.04687388, 0.95370118,
                              0.99246408, 1.00257267, 0.96993912, 1.02519882, 0.93810277,
                              0.48268112, 1.12485058, 0.84108957, 1.10272162, 0.94508517,
                              1.07946573, 0.94977558, 0.93663486, 0.98652869, 0.98037468,
                              0.87961358, 1.06685576, 1.01100015, 0.99671888, 0.95890097,
                              1.00447804, 0.92971345, 0.91349528, 1.02323946, 1.02577859,
                              0.91211533, 1.03094459, 1.11474476, 1.14168737, 1.02649575,
                              0.92977089, 0.97661125, 0.976821  , 1.12167864, 1.01383215,
                              0.98997489, 0.92519519, 1.04306992, 0.97149484, 1.32099152,
                              1.0367465 , 0.69665508, 0.91062362, 1.09735028, 0.94284757,
                              1.14077798, 0.96975988, 0.99445788, 0.97743503, 1.11935207,
                              0.99061245, 0.97508935, 1.11131188, 0.84821667, 1.02849404,
                              0.91821426, 0.53449279, 1.00486452, 1.08051941, 0.99521843,
                              0.96221783, 1.0440186 , 0.81521703, 1.0894057 , 1.11203717,
                              1.01419199, 0.8865133 , 1.0886448 , 0.91025983, 1.02400012,
                              1.00608243, 0.93732792, 1.00516116, 1.15640212, 1.05316762,
                              1.1603465 , 1.1700486 , 1.08210802, 0.97742172, 1.05970082,
                              0.96771   , 0.91606506, 1.03760101, 0.923343  , 1.04202743,
                              1.00836815, 1.04797098, 0.90173025, 1.02029034, 0.90919946,
                              1.10546591, 1.07202822, 0.93411116, 0.95355117, 0.96518171,
                              0.97356549, 1.01825159, 1.10041902, 1.0306647 , 1.00240573,
                              1.17424778, 0.84697649, 0.71144744, 1.00979486, 1.02608188,
                              1.03503633, 0.94085382, 0.89019966, 1.04144332, 1.01900468,
                              1.09237709, 1.1005501 , 1.13816495, 1.18921656, 1.07641217,
                              1.02610131, 0.94688992, 1.00916288, 0.99520259, 1.04383419,
                              0.8914629 , 0.82632783, 0.82218549, 0.95617706, 1.21362093,
                              1.01690773, 1.02284861, 0.96508569, 0.93323201, 0.85838486,
                              1.20105611, 0.77928097, 0.83495525, 0.99575798, 0.96114947,
                              1.1761569 , 0.94595579, 1.10951379, 1.11344293, 0.60714757,
                              1.03140788, 0.99712   , 1.00687262, 0.97267929, 1.07229012,
                              1.09275373, 1.09082862, 0.787288  , 1.07361439, 1.0813897 ,
                              1.04333335, 0.714547  , 1.02120617, 1.10718908, 1.05357347,
                              1.03012204, 0.9114963 , 0.93601976, 1.04242308, 1.12041318,
                              1.00101232, 1.22969326, 1.15161829, 0.99415276, 1.02143429,
                              0.94636112, 1.08114318, 0.96079992, 0.94653426, 1.02658629,
                              1.06325884, 0.86697461, 0.99407943, 0.95985893, 1.03540461,
                              1.1099155 , 1.03512613, 0.88609987, 0.88210044, 1.02402373,
                              1.07324776, 0.84010564, 1.0249757 , 0.9487804 , 0.76399759,
                              0.93586195, 1.10872997, 0.98853891, 0.99840971, 1.10876968,
                              1.03350015, 1.10303379, 1.08268589, 1.04787402, 1.01740861,
                              1.04618087, 1.16730026, 0.98164013, 1.00976541, 1.04040524,
                              1.06680264, 1.1000673 , 0.7044813 , 1.0315956 , 0.99394837,
                              1.03531513, 0.9105652 , 1.03146222, 1.06874793, 1.31614444,
                              0.90292881, 0.96053845, 1.05036673, 0.83121104, 1.17975858,
                              1.1212491 , 0.95549658, 1.04059506, 0.91478524, 0.89108609,
                              1.1266986 , 0.89964061, 0.99946036, 0.96647233, 1.04281517,
                              0.78827454, 1.15204341, 0.92453591, 0.96305923, 0.94900471,
                              1.08408539, 0.90617385, 0.98667806, 0.92399051, 1.10660771,
                              0.90871755, 1.08292277, 0.76650403, 1.01491741, 1.12331611,
                              1.02522223, 1.02110709, 1.01852973, 0.92150599, 1.07809778,
                              1.14257374, 0.98380398, 0.99664231, 1.12776558, 0.94948592,
                              0.85051449, 0.85900297, 1.05544595, 0.91412861, 1.0484394 ,
                              0.97829921, 0.98202048, 1.14047031, 0.95379987, 0.91766237,
                              1.01589703, 1.0824786 , 1.12013037, 0.88112188, 1.00863211,
                              0.92110846, 1.01895688, 1.02187464, 0.98720769, 0.85154246,
                              1.1011457 , 1.02685091, 1.09628736, 0.98466108, 1.04139473,
                              1.06269857, 1.03416093, 1.03975772, 0.98137741, 1.14413735,
                              0.96292444, 1.01744469, 1.22381866, 0.92417892, 1.06754583,
                              0.97316372, 0.93383873, 1.02873736, 0.80390796, 1.03196751,
                              0.64875596, 0.98344423, 1.1250535 , 0.87262625, 0.87091717,
                              1.13906108, 1.05110702, 0.98772745, 1.21762225, 0.98077481,
                              1.05500001, 1.05349047, 0.93513032, 0.98487021, 0.67353588,
                              1.03058237, 1.00919155, 1.13908548, 0.92211026, 0.93359272,
                              0.95502617, 0.88087176, 0.8987916 ]),
                    atol=NUMPY_atol)

    # sample_weights1
    assert_allclose(drift_explainer.sample_weights1[:5],
                    np.array([1., 1., 1., 1., 1.]),
                    atol=NUMPY_atol)

    # sample_weights2
    assert_allclose(drift_explainer.sample_weights2[:5],
                    np.array([1., 1., 1., 1., 1.]),
                    atol=NUMPY_atol)

    assert drift_explainer.cat_feature_indices == []
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
    assert drift_explainer.iteration_range == (0, 105)
    assert drift_explainer.n_features == 30
    assert drift_explainer.task == 'classification'


def test_iris_catboost_ModelDriftExplainer():
    dataset = datasets.load_iris()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
    clf = CatBoostClassifier(loss_function='MultiClass',
                             learning_rate=0.1,
                             iterations=1000,
                             max_depth=6,
                             early_stopping_rounds=20,
                             random_seed=RANDOM_SEED,
                             verbose=False)
    clf.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)])
    drift_explainer = ModelDriftExplainer(clf)
    drift_explainer.fit(X1=X_train, X2=X_test, y1=y_train, y2=y_test)

    # prediction drift "raw"
    prediction_drift_ref = [DriftMetricsNum(mean_difference=0.2556635888899628, wasserstein=0.2931763654790207, ks_test=BaseStatisticalTestResult(statistic=0.17142857142857143, pvalue=0.2821678346768163)),
                            DriftMetricsNum(mean_difference=0.2536708490324236, wasserstein=0.2625697569861726, ks_test=BaseStatisticalTestResult(statistic=0.10793650793650794, pvalue=0.8205934119780005)),
                            DriftMetricsNum(mean_difference=-0.5093344379223861, wasserstein=0.5240813705758235, ks_test=BaseStatisticalTestResult(statistic=0.18095238095238095, pvalue=0.22712915347334828))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(),
                                    prediction_drift_ref)

    # prediction drift "proba"
    prediction_drift_proba_ref = [DriftMetricsNum(mean_difference=0.06068744478888638, wasserstein=0.062073873098040216, ks_test=BaseStatisticalTestResult(statistic=0.12063492063492064, pvalue=0.7069803362151523)),
                                  DriftMetricsNum(mean_difference=0.07308349137417464, wasserstein=0.07352443840551189, ks_test=BaseStatisticalTestResult(statistic=0.12698412698412698, pvalue=0.6467769104301901)),
                                  DriftMetricsNum(mean_difference=-0.13377093616306102, wasserstein=0.13459925728761946, ks_test=BaseStatisticalTestResult(statistic=0.18095238095238095, pvalue=0.22712915347334828))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(prediction_type='proba'),
                                    prediction_drift_proba_ref)

    # target drift
    assert_drift_metrics_equal(drift_explainer.get_target_drift(),
                               DriftMetricsCat(wasserstein=0.09523809523809523,
                                               jensen_shannon=0.07382902143706498,
                                               chi2_test=Chi2TestResult(statistic=1.3333333333333333,
                                                                        pvalue=0.5134171190325922,
                                                                        dof=2,
                                                                        contingency_table=pd.DataFrame([[33.0, 34.0, 38.0], [17.0, 16.0, 12.0]],
                                                                                                       index=['X1', 'X2'], columns=[0, 1, 2]))))

    # performance_metrics_drift
    assert_performance_metrics_drift_equal(drift_explainer.get_performance_metrics_drift(),
                                           PerformanceMetricsDrift(ClassificationMetrics(accuracy=0.9904761904761905, log_loss=0.06196156247077523),
                                                                   ClassificationMetrics(accuracy=0.9333333333333333, log_loss=0.14716856908904924)))

    # tree_based_drift_values "node_size"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='node_size'),
                    np.array([[ 3.88114597],
                              [ 7.84748116],
                              [ 6.75366501],
                              [11.90863769]]),
                    atol=NUMPY_atol)

    # tree_based_drift_values "mean"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='mean'),
                    np.array([[-0.01354953, -0.00602339,  0.01957293],
                              [-0.05742398, -0.01429501, -0.06161435],
                              [ 0.14828041,  0.05454545, -0.00282586],
                              [ 0.17835669,  0.2194438 , -0.46446716]]),
                    atol=NUMPY_atol)

    # tree_based_drift_values "mean_norm"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='mean_norm'),
                    np.array([[-0.09855691, -0.09208058, -0.07090097],
                              [-0.09248865, -0.05182248, -0.0942603 ],
                              [-0.06083239, -0.14343543, -0.21452551],
                              [ 0.03280696,  0.07820104, -0.60642925]]),
                    atol=NUMPY_atol)

    # feature_drift - argument as a string
    assert_drift_metrics_equal(drift_explainer.get_feature_drift('petal width (cm)'),
                               DriftMetricsNum(mean_difference=-0.16412698412698457,
                                               wasserstein=0.16412698412698412,
                                               ks_test=BaseStatisticalTestResult(statistic=0.17142857142857143,
                                                                                 pvalue=0.2821678346768163)))

    # feature_drift - argument as integer
    assert_drift_metrics_equal(drift_explainer.get_feature_drift(2),
                               DriftMetricsNum(mean_difference=-0.2765079365079357,
                                               wasserstein=0.2777777777777778,
                                               ks_test=BaseStatisticalTestResult(statistic=0.1523809523809524,
                                                                                 pvalue=0.41885114043708227)))

    # all feature drifts
    feature_drifts_ref = [DriftMetricsNum(mean_difference=-0.18571428571428505, wasserstein=0.19968253968253974, ks_test=BaseStatisticalTestResult(statistic=0.16507936507936508, pvalue=0.3237613427576299)),
                          DriftMetricsNum(mean_difference=-0.08825396825396803, wasserstein=0.1301587301587301, ks_test=BaseStatisticalTestResult(statistic=0.14285714285714285, pvalue=0.499646880472137)),
                          DriftMetricsNum(mean_difference=-0.2765079365079357, wasserstein=0.2777777777777778, ks_test=BaseStatisticalTestResult(statistic=0.1523809523809524, pvalue=0.41885114043708227)),
                          DriftMetricsNum(mean_difference=-0.16412698412698457, wasserstein=0.16412698412698412, ks_test=BaseStatisticalTestResult(statistic=0.17142857142857143, pvalue=0.2821678346768163))]
    assert_drift_metrics_list_equal(drift_explainer.get_feature_drifts(),
                                    feature_drifts_ref)

    # tree_based_correction_weights with default params
    assert_allclose(drift_explainer.get_tree_based_correction_weights(),
                    np.array([1.46778441, 0.52294752, 0.63499851, 1.11395665, 1.51159197,
                              0.73016553, 0.70492996, 0.68508753, 1.32705675, 1.19768229,
                              1.6776906 , 1.26519898, 0.38068631, 0.68758125, 0.91815123,
                              0.84310013, 0.61681574, 1.34260229, 1.56001491, 1.443861  ,
                              0.70120618, 0.65058177, 1.78956352, 1.31837229, 0.61456506,
                              0.8542716 , 0.71941464, 0.45689895, 1.19019827, 0.63635147,
                              1.89923028, 1.09247435, 0.72878899, 0.74354047, 0.63547501,
                              1.21954902, 1.44866555, 1.04356531, 0.69035447, 1.3739053 ,
                              1.73660237, 0.33077669, 1.39210873, 1.52170161, 0.63702699,
                              1.86402732, 1.18316883, 0.40800216, 1.16027329, 0.79378744,
                              1.14638635, 1.14222676, 1.2546594 , 1.63006777, 0.59401333,
                              0.68203216, 1.52611396, 1.07152711, 1.39406928, 1.33558325,
                              0.51887858, 1.17583738, 1.23719303, 0.72742515, 0.91513678,
                              0.50291088, 0.72376554, 0.97899664, 0.84310013, 1.8191181 ,
                              0.66860961, 0.78368263, 0.31702045, 0.62884497, 1.51467351,
                              0.62243733, 0.35189584, 2.25980288, 0.67056316, 0.68112491,
                              0.7548836 , 0.71009471, 0.93523464, 0.47412222, 1.5718063 ,
                              1.49699883, 1.69508587, 0.74675178, 0.68287764, 2.04223207,
                              1.36987599, 0.61456506, 0.62397505, 0.329149  , 0.84031679,
                              0.77182041, 1.02214697, 0.93128777, 0.84400149, 1.3790245 ,
                              0.49808655, 0.78799818, 0.82521918, 1.21908353, 0.6493135 ]),
                    atol=NUMPY_atol)

    # tree_based_correction_weights with "max_depth = 1"
    assert_allclose(drift_explainer.get_tree_based_correction_weights(max_depth=1),
                    np.array([0.9674496 , 0.78989541, 0.76883768, 1.10634011, 1.10160442,
                              0.79566687, 1.14505217, 0.76883768, 1.1784106 , 1.10420233,
                              1.19467364, 1.10774096, 0.80356229, 0.85297475, 1.15812415,
                              1.15090691, 0.78878774, 1.15812415, 1.10160442, 1.10160442,
                              0.76883768, 1.10907358, 1.19467364, 1.17241594, 0.76991734,
                              1.14364165, 0.76883768, 0.75560207, 1.11460987, 0.78878774,
                              1.19467364, 1.16721319, 1.15812415, 0.76883768, 1.14471797,
                              1.11121168, 1.10420233, 1.16510963, 0.76883768, 1.05986901,
                              1.09714384, 0.82007931, 1.0726481 , 1.00569346, 0.76991734,
                              1.1784106 , 1.07040247, 0.75560207, 1.1784106 , 0.76883768,
                              1.08195813, 1.11354292, 1.1784106 , 1.07040247, 0.78878774,
                              0.76883768, 1.1784106 , 1.11354292, 1.06896562, 1.1784106 ,
                              1.14265495, 1.10651888, 1.10420233, 0.76991734, 1.15812415,
                              0.78323856, 0.76991734, 1.17106693, 1.15090691, 1.1784106 ,
                              0.76991734, 1.20509256, 0.82007931, 0.76883768, 1.10160442,
                              1.14849744, 0.91122631, 1.19467364, 0.76883768, 1.14128403,
                              0.76883768, 0.76883768, 1.02605152, 0.80356229, 1.07040247,
                              1.1784106 , 1.08195813, 0.76883768, 0.76991734, 1.1784106 ,
                              1.1784106 , 0.76991734, 1.15090691, 0.82007931, 0.76883768,
                              1.13375469, 1.1522327 , 1.09307662, 1.14522398, 1.10160442,
                              0.75560207, 0.76883768, 1.13375469, 1.02672997, 0.78878774]),
                    atol=NUMPY_atol)

    # sample_weights1
    assert_allclose(drift_explainer.sample_weights1[:5],
                    np.array([1., 1., 1., 1., 1.]),
                    atol=NUMPY_atol)

    # sample_weights2
    assert_allclose(drift_explainer.sample_weights2[:5],
                    np.array([1., 1., 1., 1., 1.]),
                    atol=NUMPY_atol)

    assert drift_explainer.cat_feature_indices == []
    assert drift_explainer.class_names == ['0', '1', '2']
    assert drift_explainer.feature_names == ['sepal length (cm)',
                                             'sepal width (cm)',
                                             'petal length (cm)',
                                             'petal width (cm)']
    assert drift_explainer.iteration_range == (0, 82)
    assert drift_explainer.n_features == 4
    assert drift_explainer.task == 'classification'
