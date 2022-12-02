import pandas as pd, numpy as np
from numpy.testing import assert_allclose
from sklearn import datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
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


def test_boston_xgboost_ModelDriftExplainer():
    boston = datasets.load_boston()
    boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
    X_train, X_test, y_train, y_test = train_test_split(boston_df, boston.target, test_size=0.3,
                                                        random_state=RANDOM_SEED)
    model = XGBRegressor(n_estimators=1000,
                         booster="gbtree",
                         objective="reg:squarederror",
                         learning_rate=0.05,
                         max_depth=6,
                         seed=RANDOM_SEED,
                         use_label_encoder=False)
    model.fit(X=X_train, y=y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20, verbose=0)
    drift_explainer = ModelDriftExplainer(model)
    drift_explainer.fit(X_train, X_test, y_train, y_test)

    # prediction drift
    prediction_drift_ref = [DriftMetricsNum(mean_difference=-0.7889487954289152, wasserstein=1.0808420273082935, ks_test=BaseStatisticalTestResult(statistic=0.052743086529884034, pvalue=0.9096081584010306))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(),
                                    prediction_drift_ref)

    # target drift
    assert_drift_metrics_equal(drift_explainer.get_target_drift(),
                               DriftMetricsNum(mean_difference=-0.609240261671129,
                                               wasserstein=1.3178114778471604,
                                               ks_test=BaseStatisticalTestResult(statistic=0.07857567647933393,
                                                                                 pvalue=0.4968030078636394)))

    # performance_metrics_drift
    assert_performance_metrics_drift_equal(drift_explainer.get_performance_metrics_drift(),
                                           PerformanceMetricsDrift(RegressionMetrics(mse=0.3643813701486243,
                                                                                     explained_variance=0.9960752192224699),
                                                                   RegressionMetrics(mse=12.419719495108291,
                                                                                     explained_variance=0.8095694395593922)))

    # tree_based_drift_values "node_size"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='node_size'),
                    np.array([[4.31765569],
                              [0.53994982],
                              [1.2299105 ],
                              [0.05466186],
                              [2.40478511],
                              [5.62793724],
                              [2.3531374 ],
                              [3.78890353],
                              [0.73237733],
                              [1.65802895],
                              [1.39955313],
                              [2.47726252],
                              [6.99832357]]),
                    atol=NUMPY_atol)

    # tree_based_drift_values "mean"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='mean'),
                    np.array([[-0.04545056],
                              [ 0.00084117],
                              [ 0.18597606],
                              [-0.00062653],
                              [ 0.24192436],
                              [-0.55327918],
                              [-0.05149254],
                              [-0.25770117],
                              [ 0.03011812],
                              [ 0.0020817 ],
                              [-0.15965889],
                              [-0.13940995],
                              [-0.04227164]]),
                    atol=NUMPY_atol)

    # tree_based_drift_values "mean_norm"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='mean_norm'),
                    np.array([[ 0.05234698],
                              [ 0.0058806 ],
                              [ 0.00328171],
                              [-0.00030556],
                              [ 0.07280695],
                              [-0.29861327],
                              [-0.00603697],
                              [-0.2603778 ],
                              [ 0.01561826],
                              [-0.02779742],
                              [-0.01725041],
                              [-0.03128068],
                              [-0.03379194]]),
                    atol=NUMPY_atol)

    # feature_drift_LSTAT
    assert_drift_metrics_equal(drift_explainer.get_feature_drift('LSTAT'),
                               DriftMetricsNum(mean_difference=0.7378638864109419,
                                                                         wasserstein=0.8023078352661315,
                                                                         ks_test=BaseStatisticalTestResult(statistic=0.08887154326494201,
                                                                                                           pvalue=0.3452770147763923)))

    # feature_drift_feature_0
    assert_drift_metrics_equal(drift_explainer.get_feature_drift(0),
                               DriftMetricsNum(mean_difference=-1.1253475613291695,
                                                                   wasserstein=1.1305975918079103,
                                                                   ks_test=BaseStatisticalTestResult(statistic=0.0618123699078204,
                                                                                                     pvalue=0.7813257636577198)))

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
                    np.array([1.16419732, 1.10087397, 1.21368666, 1.13727639, 1.13989036,
                              1.18977695, 1.01215575, 1.10059681, 0.72558209, 1.00675234,
                              1.10666682, 0.88555298, 1.12977301, 1.16608115, 0.91400419,
                              0.44987693, 1.04860138, 1.15972868, 0.74749602, 0.78955673,
                              1.23054107, 1.0929581 , 1.15461204, 0.93743351, 0.89136442,
                              1.18001834, 1.09512827, 1.20852801, 1.07037641, 0.85541822,
                              0.71586788, 1.10300282, 1.25814986, 0.96320127, 0.96420466,
                              1.01631317, 1.31017151, 0.99762216, 0.91570282, 1.23407839,
                              1.01356292, 1.03317556, 0.98659978, 1.14176654, 0.8256704 ,
                              1.05326507, 1.15819153, 0.96044076, 1.04398531, 0.97956093,
                              1.13806087, 0.92484557, 0.60986512, 0.91123648, 1.06589494,
                              0.84968641, 1.1733163 , 0.93350917, 0.88924206, 1.05747077,
                              0.93490889, 0.79767375, 0.92145929, 0.37230812, 1.10293867,
                              1.09235481, 1.37000448, 0.89159463, 1.05314073, 1.04337542,
                              1.17859982, 1.05619742, 1.19299521, 0.49073022, 1.12006103,
                              1.10935394, 1.20617108, 0.98542949, 1.19876176, 0.49210116,
                              0.76742674, 1.06288453, 0.98133668, 0.79451027, 1.05343443,
                              1.17409533, 1.0322569 , 0.84641052, 0.72454594, 1.0972863 ,
                              1.11972538, 1.0166767 , 1.12840246, 0.90672729, 0.70155724,
                              1.18006586, 0.92539464, 0.32431158, 1.04676889, 1.07684215,
                              1.12142957, 1.12868632, 0.80044461, 1.11108876, 1.07978027,
                              0.83810027, 0.97369048, 0.94380991, 1.06643724, 1.03637159,
                              1.09826332, 1.21987827, 1.1017478 , 0.92381585, 1.05592795,
                              0.77292046, 1.26144311, 1.11049506, 0.50631903, 0.69105845,
                              1.05058848, 0.92704126, 0.59578928, 1.09463525, 1.01638428,
                              0.96861518, 1.06710481, 1.0882454 , 0.69635315, 0.84551532,
                              1.16587031, 0.91786233, 1.21218501, 1.11080734, 0.86970289,
                              1.09885839, 1.23491866, 0.61032684, 0.94564681, 0.94738328,
                              1.059863  , 1.13433497, 0.83917301, 1.15907066, 0.84877355,
                              0.33947135, 0.49498701, 0.88139004, 0.99945658, 0.95302861,
                              1.48348648, 1.09300884, 1.11657693, 1.13850024, 0.33491379,
                              0.50668992, 1.24825669, 1.13978931, 1.20383783, 0.94804158,
                              0.99665375, 1.22177126, 1.0945194 , 1.02675212, 1.30485921,
                              0.58660552, 0.55305576, 1.17434394, 1.17382957, 1.1202571 ,
                              0.96642816, 1.01531836, 1.10319784, 0.86347228, 1.00810461,
                              0.56946217, 1.08268008, 0.86923329, 1.0317189 , 0.93847969,
                              0.845586  , 0.97831139, 1.00289034, 1.14807371, 0.86903538,
                              1.13722926, 1.1006587 , 1.16877497, 1.01227611, 1.07845423,
                              1.11708513, 1.19417547, 1.2778751 , 0.98797798, 1.07728855,
                              1.05499927, 1.04710719, 1.08808864, 0.96846122, 1.32186046,
                              0.8530053 , 1.04979525, 1.09932135, 0.7429571 , 0.92881187,
                              1.30223646, 0.42061162, 1.19721629, 0.73952189, 1.04644862,
                              1.17597349, 1.0840867 , 0.74224707, 1.13574442, 0.94866608,
                              0.95590973, 0.94671777, 0.9254706 , 0.9839863 , 1.0244442 ,
                              0.77062547, 0.84602691, 0.9576317 , 0.88120068, 0.89514653,
                              1.15687421, 1.05989279, 1.13865161, 1.04575364, 1.11247464,
                              1.17972616, 0.95284698, 0.99251302, 0.98733136, 1.10005399,
                              1.28689456, 1.00010831, 1.28234576, 0.89869426, 1.11778495,
                              1.05600081, 1.12962457, 1.03721409, 1.07050115, 1.31205803,
                              1.22222613, 1.1449303 , 0.94381272, 0.47570943, 0.98423073,
                              0.57225095, 1.09066898, 0.96422086, 0.89040329, 0.32512642,
                              1.02420319, 1.32294277, 1.08759698, 1.10158941, 0.61790862,
                              0.96842455, 1.13527867, 0.85848503, 1.1100173 , 0.86949728,
                              0.81562783, 1.15702292, 0.84311207, 1.12182998, 1.06690836,
                              0.76243479, 0.92992144, 0.93082927, 0.51646473, 0.75581889,
                              1.13794611, 0.88634101, 1.01195516, 1.0558324 , 1.05306641,
                              1.08135681, 1.06816819, 1.09851691, 1.33354986, 1.13348136,
                              1.09352752, 1.01020482, 1.16867974, 1.10760027, 0.95324276,
                              0.9707388 , 1.1306543 , 1.1445239 , 0.96796771, 0.96245229,
                              1.2453499 , 0.9909731 , 0.64256786, 1.0459765 , 1.05809618,
                              0.92863222, 1.04914003, 0.94390974, 0.88043042, 1.12183045,
                              1.21387541, 1.06464791, 0.9586424 , 0.95162604, 1.14590447,
                              1.11224052, 0.76397902, 1.09903285, 1.02689233, 1.02270401,
                              1.06316529, 1.09481799, 1.05671481, 0.92147167, 1.14338271,
                              0.97165919, 1.05255944, 0.66238535, 1.11540372, 1.10107548,
                              0.84405474, 1.11217045, 0.77474588, 1.01380398, 0.83058349,
                              1.16788829, 0.85915848, 0.75478956, 1.10948121, 1.03640738,
                              1.16700158, 1.07234547, 0.91683386, 0.931275  , 1.06521399,
                              0.73163952, 1.05447144, 1.14416767, 1.07729212, 1.18850243,
                              1.10836293, 1.1817005 , 1.1238247 , 1.09186981, 1.17053314,
                              0.99844788, 1.01820715, 1.25297799, 1.09458313]),
                    atol=NUMPY_atol)

    # tree_based_correction_weights with "max_depth = 2"
    assert_allclose(drift_explainer.get_tree_based_correction_weights(max_depth=2),
                    np.array([1.06705916, 1.00782503, 1.03194615, 1.01143602, 1.04150287,
                              1.00490137, 1.00716356, 1.01372945, 1.01984372, 1.0415846 ,
                              1.03286222, 0.99840509, 1.06628184, 1.03063623, 0.96939166,
                              0.8038806 , 1.03194615, 1.06036313, 0.98854907, 0.97907688,
                              1.013492  , 0.88872912, 1.05886575, 1.02508135, 1.04243298,
                              1.02569367, 1.01785115, 1.02850978, 0.9878603 , 0.96039109,
                              1.05294481, 1.05615545, 1.05709789, 1.02261963, 1.05662473,
                              1.00474926, 1.03737369, 1.01897035, 1.01220578, 1.06800669,
                              1.05100332, 1.01640816, 1.02526032, 1.05518344, 0.99446514,
                              1.00321176, 1.03888114, 1.04169193, 1.01280588, 1.0248593 ,
                              1.0249929 , 1.02327516, 0.80835604, 0.92040462, 1.01850082,
                              1.02153529, 0.90401362, 1.02508135, 1.04124058, 1.02249592,
                              1.00350377, 1.05380244, 1.00418732, 0.71625784, 1.03046312,
                              1.06836601, 1.06196184, 1.03166289, 1.00493671, 1.00978167,
                              1.00664191, 1.04896893, 1.03335223, 0.80286263, 1.01784813,
                              1.0230165 , 1.03333042, 0.84973661, 1.03394422, 0.76627792,
                              1.05628382, 1.01554171, 1.01177573, 0.7684792 , 1.00713751,
                              1.02989149, 1.05380244, 1.00484896, 0.76382491, 1.01011393,
                              0.95893291, 1.01303789, 1.05380244, 0.90396531, 0.80832806,
                              1.03219578, 1.05623757, 0.70414909, 1.03159602, 1.01011393,
                              1.02227181, 1.02247586, 1.04243298, 1.03888114, 1.03306075,
                              0.97684241, 1.01143497, 1.00390841, 1.01690221, 1.023098  ,
                              1.05092364, 1.02605504, 1.03239116, 1.04243298, 1.03159602,
                              1.00770838, 1.01753144, 1.06355041, 0.80286263, 0.75878782,
                              1.00042124, 0.86210717, 0.81805437, 1.01595349, 1.01736054,
                              1.0028074 , 1.01347956, 1.02344056, 0.81918853, 1.02189   ,
                              1.027844  , 1.06352663, 1.03735179, 1.0109767 , 1.06018503,
                              1.02446896, 1.04069356, 0.93041787, 0.92321909, 1.02691114,
                              1.0261104 , 1.05804732, 1.00032953, 1.04715949, 1.01143497,
                              0.64937092, 0.75612285, 1.02438357, 1.01978483, 1.02372891,
                              1.06364573, 1.0249929 , 1.02795647, 1.02206413, 0.62444111,
                              0.81441905, 1.04560518, 1.02249592, 1.0202202 , 1.04272037,
                              0.85373068, 1.05937769, 1.01905612, 1.02037054, 1.05917458,
                              0.70315177, 0.84265141, 1.03564249, 1.03158103, 1.05625255,
                              1.01155734, 1.02473125, 1.0248593 , 1.03110024, 1.0105692 ,
                              0.76655379, 1.01929869, 1.00111594, 1.03218851, 1.05161489,
                              1.05380244, 0.9806052 , 1.00321176, 1.03293578, 1.00571659,
                              1.08229672, 1.0087436 , 1.08268708, 1.04243298, 1.01435934,
                              1.03569911, 1.01924758, 1.07414099, 1.04461879, 1.01836087,
                              1.02202067, 1.05894745, 1.03006066, 1.02643593, 1.05709789,
                              1.01168096, 1.06689953, 1.02196073, 0.83395358, 1.00043922,
                              1.15795501, 0.71625784, 1.03587309, 0.75878782, 1.05380244,
                              1.05484844, 1.04243298, 0.84244514, 1.03735179, 1.02792702,
                              1.02966407, 1.03335223, 0.96005168, 1.02455794, 1.01088373,
                              1.04243298, 1.04050496, 1.02249592, 1.02036971, 1.0415846 ,
                              1.05424294, 1.03218851, 1.01418897, 1.03564249, 1.01220578,
                              1.01143497, 0.8492266 , 1.02931471, 1.01486963, 1.03774678,
                              1.05063759, 1.01603176, 1.06950223, 1.02451342, 1.01562316,
                              1.02864778, 1.0332124 , 1.01418897, 1.02793579, 1.03888114,
                              1.05004129, 1.01177573, 1.00748309, 0.94426584, 1.01767718,
                              0.86049152, 1.01984239, 1.03821867, 1.05709789, 0.70139784,
                              1.0416767 , 1.05709789, 1.03913362, 1.00816457, 0.76974082,
                              1.0415846 , 1.04090503, 1.04243298, 1.04440392, 1.01973511,
                              1.03564249, 1.02943286, 1.02796848, 1.0157813 , 1.02045684,
                              1.02864778, 0.8525487 , 1.00884064, 0.74854892, 1.01284357,
                              1.03997235, 1.02153529, 1.00782503, 1.03267649, 1.02778381,
                              1.01143497, 1.02772656, 1.02722247, 1.03643174, 1.03194615,
                              1.04763263, 1.04621989, 1.02569346, 1.05618285, 1.04243298,
                              1.04243298, 1.01450463, 1.01649063, 0.89104104, 1.05294481,
                              1.03799457, 1.01188382, 0.9310142 , 1.01736054, 1.01340682,
                              1.03173974, 1.01418897, 1.02446896, 1.00350377, 1.03888114,
                              1.02809208, 1.01160488, 1.00987981, 1.03888114, 1.02291403,
                              1.00799433, 1.0415846 , 1.03267649, 1.01177573, 0.9816398 ,
                              0.96714353, 1.03913362, 1.01143497, 1.00839094, 1.02525736,
                              1.01476836, 1.03806175, 1.02312352, 1.04720395, 1.03598462,
                              0.99556151, 0.90878994, 0.98549045, 0.89692268, 1.04950095,
                              1.03267649, 1.01143497, 0.87793868, 1.02596249, 1.03568617,
                              1.03888114, 1.0181819 , 0.99735705, 1.00522002, 1.0416767 ,
                              1.01143497, 1.02867809, 1.01459792, 0.8948014 , 1.05592543,
                              1.01303789, 1.03286268, 1.03194615, 1.0187867 , 1.02966407,
                              1.03347725, 0.86450652, 1.07528078, 1.00782503]),
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
    assert drift_explainer.iteration_range == (0, 143)
    assert drift_explainer.n_features == 13
    assert drift_explainer.task == 'regression'


def test_breast_cancer_xgboost_ModelDriftExplainer():
    dataset = datasets.load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
    clf = XGBClassifier(n_estimators=1000,
                        booster="gbtree",
                        objective="binary:logistic",
                        learning_rate=0.05,
                        max_depth=6,
                        use_label_encoder=False,
                        seed=RANDOM_SEED)
    clf.fit(X=X_train, y=y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=20, verbose=0)
    drift_explainer = ModelDriftExplainer(clf)
    drift_explainer.fit(X1=X_train, X2=X_valid, y1=y_train, y2=y_valid)

    # prediction drift "raw"
    prediction_drift_ref = [DriftMetricsNum(mean_difference=0.005498574879272855, wasserstein=0.3764544601013494, ks_test=BaseStatisticalTestResult(statistic=0.08323782655969908, pvalue=0.3542889176877513))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(),
                                    prediction_drift_ref)

    # prediction drift "proba"
    prediction_drift_proba_ref = [DriftMetricsNum(mean_difference=0.00798452225707491, wasserstein=0.024082025832758043, ks_test=BaseStatisticalTestResult(statistic=0.08323782655969908, pvalue=0.3542889176877513))]
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
                                           PerformanceMetricsDrift(ClassificationMetrics(accuracy=1.0, log_loss=0.016039305599991362),
                                                                   ClassificationMetrics(accuracy=0.9473684210526315, log_loss=0.11116574995208815)))

    # tree_based_drift_values "node_size"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='node_size'),
                    np.array([[0.1263785 ],
                              [0.86245768],
                              [0.03015075],
                              [0.5015792 ],
                              [0.345185  ],
                              [0.01076813],
                              [0.48651996],
                              [0.5553289 ],
                              [0.00502513],
                              [0.02160804],
                              [0.05284761],
                              [0.16747726],
                              [0.01615219],
                              [0.21016901],
                              [0.06186914],
                              [0.61067109],
                              [0.13051384],
                              [0.00904523],
                              [0.1728764 ],
                              [0.0583929 ],
                              [0.970708  ],
                              [0.78384477],
                              [1.04602337],
                              [1.23239211],
                              [1.55243827],
                              [0.09348557],
                              [0.55188395],
                              [1.57008003],
                              [0.27572543],
                              [0.05313921]]),
                    atol=NUMPY_atol)

    # tree_based_drift_values "mean"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='mean'),
                    np.array([[-9.33158904e-03],
                              [ 1.15870292e-02],
                              [ 2.25198024e-03],
                              [-5.45754250e-03],
                              [ 5.29567213e-03],
                              [ 1.24361599e-03],
                              [ 2.21942302e-02],
                              [-3.17637413e-03],
                              [ 9.47460649e-06],
                              [ 4.13937125e-03],
                              [ 1.80148733e-03],
                              [-1.81682108e-03],
                              [ 7.62334093e-05],
                              [ 1.50764895e-03],
                              [-2.89777474e-03],
                              [-1.19719725e-02],
                              [ 4.80747869e-03],
                              [ 1.99959485e-03],
                              [-4.34269903e-04],
                              [ 1.51119076e-03],
                              [ 2.62156736e-03],
                              [-9.64497653e-04],
                              [-1.93085931e-02],
                              [-2.76468329e-03],
                              [ 1.09498474e-02],
                              [-5.65097244e-03],
                              [ 1.16655661e-03],
                              [-1.93422930e-03],
                              [ 8.26090050e-04],
                              [-2.78114137e-03]]),
                    atol=NUMPY_atol)

    # tree_based_drift_values "mean_norm"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='mean_norm'),
                    np.array([[ 0.00128017],
                              [-0.00862285],
                              [ 0.00227804],
                              [-0.00702226],
                              [ 0.01434936],
                              [ 0.00044783],
                              [ 0.01191792],
                              [ 0.00368157],
                              [-0.0001583 ],
                              [-0.00016892],
                              [ 0.00055728],
                              [ 0.00208376],
                              [-0.00037917],
                              [-0.00193379],
                              [-0.00055455],
                              [-0.0112573 ],
                              [ 0.00458787],
                              [-0.00033537],
                              [ 0.00311544],
                              [-0.00054474],
                              [-0.00204518],
                              [ 0.00557142],
                              [-0.01681621],
                              [-0.02567387],
                              [ 0.01606102],
                              [-0.00331497],
                              [ 0.00925461],
                              [-0.00053606],
                              [ 0.00542589],
                              [-0.00056559]]),
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
                    np.array([1.09688861, 0.9762864 , 1.00014947, 1.01476245, 1.01354627,
                              0.9767973 , 1.04556855, 0.85788803, 0.90761728, 0.95980924,
                              1.46309695, 1.01011298, 0.96974536, 1.05906244, 0.97232291,
                              0.99180543, 0.98516202, 1.00942547, 1.26310403, 0.9748277 ,
                              1.01451781, 0.97868663, 0.97123145, 0.9733829 , 1.20145142,
                              0.9586891 , 0.97539291, 0.9768325 , 0.97842638, 0.98211348,
                              1.02099763, 0.95255366, 0.95910583, 0.97749069, 0.91711393,
                              1.0435607 , 0.92132381, 0.9767973 , 0.84949809, 1.08825167,
                              0.99230282, 0.97842638, 0.94620344, 0.99724424, 0.9036093 ,
                              0.98324182, 1.01734828, 0.96964598, 1.06762684, 0.96974536,
                              0.9949011 , 1.2057241 , 1.01011298, 0.99340268, 0.9664361 ,
                              0.97450892, 0.97488189, 0.9762864 , 1.01691498, 0.98512894,
                              1.00555434, 0.96905592, 0.96974536, 0.91122304, 0.93102404,
                              0.96065739, 0.92940275, 0.98132391, 1.12690028, 0.96900604,
                              0.94353279, 0.9762864 , 0.94343761, 1.0492391 , 0.93949154,
                              0.9934877 , 0.96645436, 1.03202716, 0.93938445, 0.92132381,
                              1.11859089, 1.02045142, 1.09223617, 0.98554884, 1.00350732,
                              0.94494111, 0.90379319, 0.90379319, 1.06169248, 0.96778998,
                              1.15590687, 1.01416578, 0.90558469, 1.00045202, 0.86778472,
                              1.07007739, 1.12243864, 0.98477793, 0.98689794, 1.10220873,
                              0.98352155, 0.9797077 , 0.95116113, 1.12059577, 0.92132381,
                              0.98599024, 0.95152545, 0.93661173, 0.97842638, 1.00527655,
                              1.19862131, 0.9949011 , 1.01011298, 1.00984077, 0.93130539,
                              0.92132381, 0.96974536, 1.05286115, 1.01476245, 0.9767973 ,
                              0.92061277, 0.90067896, 0.9949011 , 0.9767973 , 1.39335122,
                              0.94681491, 0.94807998, 0.93084373, 0.95346645, 0.96806721,
                              1.05752417, 0.97923458, 0.9207142 , 1.03982842, 1.03194528,
                              0.96923814, 0.96976656, 1.11572012, 0.91239922, 0.99233025,
                              1.15136868, 0.89028972, 0.96380939, 1.10862893, 0.98877476,
                              0.90374205, 0.97767417, 0.96089942, 1.04194554, 1.06896318,
                              0.9818003 , 0.91027932, 1.03095402, 1.09436284, 0.93486541,
                              1.01214388, 0.91603461, 0.89441691, 1.15426558, 0.93306278,
                              1.21696406, 1.07087497, 0.98775857, 0.9748277 , 1.18333091,
                              0.9311599 , 1.01432607, 0.96806721, 0.96493244, 0.94681491,
                              0.9767973 , 1.02117961, 0.9818436 , 0.97395557, 0.90716718,
                              1.00951905, 1.11828809, 0.98335911, 0.88741634, 0.94236893,
                              0.98855728, 0.9610783 , 1.00511946, 0.9610783 , 1.05466407,
                              0.99150881, 0.89978799, 0.92824642, 0.9733829 , 0.98657956,
                              0.9767973 , 0.95808787, 0.96570979, 0.9767973 , 0.98211348,
                              1.00552185, 1.01538974, 1.07342214, 1.24891905, 0.97457285,
                              0.9818003 , 0.97151205, 1.05201347, 1.02833475, 0.96964598,
                              1.04067146, 1.29794378, 1.01433355, 1.11445973, 1.17866278,
                              0.97602717, 0.97940093, 0.96672796, 0.9036093 , 1.04055059,
                              1.08716264, 0.91713342, 0.97842638, 0.97929098, 0.91711393,
                              1.07971923, 0.94648404, 1.06588569, 0.93929923, 0.86900744,
                              0.9733829 , 0.99358489, 0.93222083, 0.981091  , 0.98941224,
                              1.14817263, 1.00549738, 0.9264055 , 0.9748277 , 1.00375663,
                              0.99860652, 1.03187757, 1.00944462, 1.01416578, 1.19213111,
                              0.9861027 , 1.12463125, 0.98891136, 0.9488511 , 1.18800708,
                              0.93474089, 1.15864599, 0.99223952, 0.9513338 , 0.99024567,
                              1.16595902, 0.97767417, 1.02270935, 0.91477253, 0.9767973 ,
                              1.15087793, 0.94273799, 0.95093785, 0.96452201, 1.02728696,
                              1.01577319, 0.97395557, 0.97675389, 1.03329837, 0.95444102,
                              1.24367776, 0.93817417, 0.91544856, 0.99685911, 0.96439299,
                              0.95498178, 1.01416578, 0.96520215, 1.00139353, 1.00549738,
                              0.97190444, 1.00996603, 1.00851361, 1.02635158, 0.99412489,
                              0.92994705, 1.1962142 , 1.00936136, 0.92669589, 1.00291292,
                              1.02880622, 0.9610783 , 0.89966342, 0.93919222, 0.94716181,
                              1.00306662, 0.97671976, 0.97785108, 0.99097539, 0.96974536,
                              1.04667349, 0.97762583, 0.99467277, 0.93224644, 1.0239402 ,
                              1.00218875, 0.9767973 , 0.97933463, 0.93395503, 0.98953181,
                              0.96951773, 1.04147533, 0.97395557, 0.9767973 , 1.05049822,
                              0.85858325, 1.00839429, 0.95997819, 0.91703499, 0.9764657 ,
                              1.02576597, 1.05536556, 0.95116113, 0.9767973 , 1.00944462,
                              0.99493695, 1.00070629, 0.89818716, 0.97842638, 1.15284526,
                              0.9926688 , 0.97577954, 0.97078205, 0.99270135, 1.10890484,
                              1.0400484 , 0.9949011 , 0.96735107, 0.99550516, 0.98019335,
                              0.89467117, 0.85698742, 1.04524239, 0.91764191, 0.99135506,
                              0.98274187, 1.13238986, 1.04141147, 0.95901436, 0.96155602,
                              0.97395557, 1.116924  , 1.01011298, 1.38139845, 1.06746572,
                              1.23827099, 0.9767973 , 0.9782942 , 0.91954068, 0.9436986 ,
                              1.00983344, 0.97842638, 0.96974536, 0.99754785, 1.0792625 ,
                              0.9782942 , 0.9347743 , 1.07728331, 0.98969799, 0.96230798,
                              1.00826171, 0.97846367, 1.0976199 , 0.87795576, 0.98224721,
                              0.9818003 , 0.888337  , 0.95865103, 0.90196546, 0.99737843,
                              0.85545711, 0.97395557, 1.09080592, 0.88192188, 0.90172028,
                              1.10812858, 1.02735467, 0.97096186, 1.11491658, 1.00854637,
                              1.01446387, 1.00485001, 0.89966342, 1.10834215, 1.07745083,
                              0.9554235 , 1.02499975, 1.05634889, 1.04846522, 1.07746242,
                              0.99881543, 0.98052599, 0.99233025]),
                    atol=NUMPY_atol)

    # tree_based_correction_weights with "max_depth = 1"
    assert_allclose(drift_explainer.get_tree_based_correction_weights(max_depth=1),
                    np.array([0.99060717, 1.00097782, 1.01195055, 0.99190102, 0.99296686,
                              1.00097782, 1.02115759, 0.95965535, 0.99190102, 0.99190102,
                              1.03780921, 0.99190102, 1.00097782, 1.02115759, 1.00097782,
                              1.00097782, 0.99190102, 0.99190102, 1.03818753, 1.00097782,
                              0.99190102, 0.99190102, 1.00097782, 0.99190102, 1.02609205,
                              1.01195055, 1.00097782, 1.00097782, 0.99190102, 0.98308249,
                              0.99190102, 0.98862601, 1.00097782, 0.99190102, 0.99190102,
                              1.00097782, 0.99190102, 1.00097782, 0.98176558, 1.02377208,
                              0.99296686, 0.99190102, 0.98964166, 0.99190102, 0.98308249,
                              0.98176558, 1.00097782, 1.00097782, 1.02115759, 1.00097782,
                              1.00097782, 1.03348572, 0.99190102, 1.00295377, 0.98308249,
                              1.00097782, 1.00097782, 1.00097782, 1.00097782, 0.99190102,
                              1.01977884, 1.00097782, 1.00097782, 0.98308249, 0.98308249,
                              0.99190102, 1.00913713, 0.99190102, 1.01802648, 1.00232049,
                              0.98964166, 1.00097782, 0.99296686, 1.02115759, 1.01195055,
                              0.98176558, 1.00097782, 1.02115759, 0.99190102, 0.99190102,
                              1.01693375, 1.00097782, 1.01350404, 0.99190102, 0.97093225,
                              0.99070507, 0.99190102, 0.99190102, 1.00778359, 1.00097782,
                              1.03391585, 0.99190102, 0.98960065, 1.00097782, 0.98308249,
                              1.02115759, 1.01514945, 1.00097782, 0.99190102, 1.01345846,
                              1.00407111, 1.00097782, 0.99190102, 1.02115759, 0.99190102,
                              0.98308249, 0.99296686, 0.98308249, 0.99190102, 1.01195055,
                              1.02400318, 1.00097782, 0.99190102, 0.99190102, 0.9900491 ,
                              0.99190102, 1.00097782, 1.01011723, 0.99190102, 1.00097782,
                              0.99190102, 0.99190102, 1.00097782, 1.00097782, 1.02609205,
                              0.99190102, 0.96963163, 0.98862601, 0.99190102, 1.00097782,
                              1.01693375, 1.01195055, 0.99190102, 0.99193593, 1.02115759,
                              1.00097782, 1.00097782, 1.02400318, 0.98308249, 0.99190102,
                              1.02719463, 0.96867101, 1.00097782, 1.02115759, 1.01454525,
                              0.99190102, 0.99190102, 0.99990338, 1.02115759, 1.02115759,
                              0.99190102, 0.98308249, 1.00097782, 1.03710965, 1.02477633,
                              1.00097782, 0.99190102, 0.99296686, 1.02643535, 0.99190102,
                              1.0377968 , 1.02115759, 1.00295377, 1.00097782, 1.0215352 ,
                              0.98968833, 1.00097782, 1.00097782, 1.00097782, 0.99190102,
                              1.00097782, 0.99190102, 1.00097782, 0.99190102, 0.99190102,
                              0.99190102, 1.02115759, 0.99190102, 0.98308249, 0.98964166,
                              1.01095792, 1.00097782, 1.02115759, 1.00097782, 1.00093062,
                              0.99190102, 0.99190102, 0.98176558, 0.99190102, 0.99190102,
                              1.00097782, 1.00097782, 1.00083388, 1.00097782, 0.98308249,
                              1.00097782, 0.99190102, 1.02115759, 1.02443358, 1.00097782,
                              0.99190102, 0.99990338, 1.00083388, 1.0229736 , 1.00097782,
                              1.00305635, 1.02380811, 0.99831398, 1.02487216, 1.02238979,
                              1.00097782, 0.99190102, 1.00097782, 0.98308249, 1.00124462,
                              1.02115759, 0.99070507, 0.99190102, 0.99190102, 0.99190102,
                              1.02115759, 0.99990338, 1.02115759, 0.98743885, 0.98175514,
                              0.99190102, 1.01195055, 0.98308249, 0.99190102, 0.99190102,
                              1.00300905, 0.99190102, 1.00207785, 1.00097782, 0.99190102,
                              1.01195055, 1.02643535, 0.99190102, 0.99190102, 1.01331097,
                              0.99190102, 0.98964166, 1.02106398, 0.99096913, 1.01350404,
                              0.99190102, 1.02609205, 0.99190102, 0.99990338, 0.99190102,
                              1.03639558, 0.99190102, 0.98308249, 0.98964166, 1.00097782,
                              1.02719463, 0.99190102, 0.99930399, 0.99070507, 1.01195055,
                              0.99190102, 0.99190102, 0.98968833, 1.00097782, 0.99190102,
                              1.02506037, 0.97603687, 0.99190102, 1.00097782, 1.00097782,
                              0.99190102, 0.99190102, 0.99296686, 1.01207895, 0.99190102,
                              0.99296686, 0.99190102, 0.99190102, 1.01195055, 1.00295377,
                              0.99190102, 1.03776028, 1.00097782, 0.99190102, 1.00097782,
                              0.99190102, 1.00097782, 0.99190102, 0.98862601, 0.99990338,
                              1.00097782, 0.99070507, 0.99190102, 1.00097782, 1.00097782,
                              1.00166545, 0.98308249, 1.00097782, 0.98743885, 0.99190102,
                              0.99190102, 1.00097782, 0.99190102, 0.98964166, 0.98176558,
                              0.99990338, 1.00310274, 0.99190102, 1.00097782, 1.02115759,
                              1.01207895, 0.99190102, 1.00295377, 0.98308249, 0.99190102,
                              1.01207895, 0.99060717, 0.99190102, 1.00097782, 0.99190102,
                              1.00097782, 0.99190102, 0.98308249, 0.99190102, 1.03545037,
                              1.00026788, 1.01195055, 1.00097782, 0.99190102, 1.02715847,
                              1.02115759, 1.00097782, 0.99990338, 0.99190102, 0.99162617,
                              0.98308249, 0.98308249, 1.02115759, 0.97603687, 0.99190102,
                              0.99190102, 1.01331097, 1.01195055, 0.97972288, 0.99190102,
                              0.99190102, 1.02115759, 0.99190102, 1.02747846, 1.02377208,
                              1.03353577, 1.00097782, 1.00097782, 0.99190102, 0.99203397,
                              0.99190102, 0.99190102, 1.00097782, 0.99190102, 1.03035393,
                              1.00097782, 0.99190102, 1.02119353, 1.00097782, 1.00097782,
                              0.98308249, 0.99990338, 1.02377208, 0.99190102, 1.00097782,
                              0.99190102, 0.98308249, 0.99190102, 0.98743885, 1.01195055,
                              0.97603687, 0.99190102, 1.02115759, 0.99190102, 0.99059663,
                              1.02377208, 1.00097782, 1.00134271, 1.02115759, 1.00097782,
                              0.99190102, 1.00097782, 0.99190102, 1.0013055 , 0.9577269 ,
                              0.99190102, 1.00097782, 1.02115759, 1.00077649, 1.01928347,
                              0.99070507, 0.98308249, 0.99190102]),
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
    assert drift_explainer.iteration_range == (0, 126)
    assert drift_explainer.n_features == 30
    assert drift_explainer.task == 'classification'


def test_iris_xgboost_XGBClassifier():
    dataset = datasets.load_iris()
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
    drift_explainer = ModelDriftExplainer(clf)
    drift_explainer.fit(X1=X_train, X2=X_test, y1=y_train, y2=y_test)

    # prediction drift "raw"
    prediction_drift_ref = [DriftMetricsNum(mean_difference=0.31093458145383807, wasserstein=0.310934581453838, ks_test=BaseStatisticalTestResult(statistic=0.06349206349206349, pvalue=0.9987212484986797)),
                            DriftMetricsNum(mean_difference=0.3232848411632908, wasserstein=0.3318073130907522, ks_test=BaseStatisticalTestResult(statistic=0.12698412698412698, pvalue=0.6467769104301901)),
                            DriftMetricsNum(mean_difference=-0.5564053781212321, wasserstein=0.5568392310587188, ks_test=BaseStatisticalTestResult(statistic=0.17142857142857143, pvalue=0.2821678346768163))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(),
                                    prediction_drift_ref)

    # prediction drift "proba"
    prediction_drift_proba_ref = [DriftMetricsNum(mean_difference=0.06122570359665486, wasserstein=0.06138405880136859, ks_test=BaseStatisticalTestResult(statistic=0.1111111111111111, pvalue=0.793799989988573)),
                                  DriftMetricsNum(mean_difference=0.08154049228640303, wasserstein=0.08205600790975118, ks_test=BaseStatisticalTestResult(statistic=0.12698412698412698, pvalue=0.6467769104301901)),
                                  DriftMetricsNum(mean_difference=-0.1427661934187488, wasserstein=0.14276781702443725, ks_test=BaseStatisticalTestResult(statistic=0.19047619047619047, pvalue=0.1805850065949114))]
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
                                           PerformanceMetricsDrift(ClassificationMetrics(accuracy=1.0, log_loss=0.045063073312242824),
                                                                   ClassificationMetrics(accuracy=0.9333333333333333, log_loss=0.16192325585418277)))

    # tree_based_drift_values "node_size"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='node_size'),
                    np.array([[ 1.96002375],
                              [ 4.03749391],
                              [16.37810068],
                              [24.55940758]]),
                    atol=NUMPY_atol)

    # tree_based_drift_values "mean"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='mean'),
                    np.array([[ 0.        ,  0.18063744, -0.0278192 ],
                              [ 0.        , -0.00393914, -0.06837239],
                              [ 0.31093455,  0.03153828, -0.26869759],
                              [ 0.        ,  0.11504829, -0.19151622]]),
                    atol=NUMPY_atol)

    # tree_based_drift_values "mean_norm"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='mean_norm'),
                    np.array([[ 0.        ,  0.01680469,  0.00507903],
                              [ 0.        , -0.00718861,  0.02268815],
                              [ 0.31093455, -0.12144406,  0.0417556 ],
                              [ 0.        ,  0.06756211, -0.16671063]]),
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
                    np.array([1.52938066, 0.70303783, 0.69585122, 1.14692382, 1.14698728,
                              0.72033464, 1.22632416, 0.72033464, 1.22632416, 1.142113  ,
                              1.23736756, 1.142113  , 0.56368673, 0.86018518, 1.22632416,
                              1.22632416, 0.7208303 , 1.22632416, 1.14698728, 1.14698728,
                              0.69775839, 1.04277591, 1.23736756, 1.22632416, 0.70428396,
                              1.13630004, 0.7208303 , 0.69775839, 1.142113  , 0.73174406,
                              1.23736756, 1.22632416, 1.22632416, 0.69785836, 1.19780314,
                              1.15211681, 1.14698728, 1.22632416, 0.69585122, 1.03271783,
                              1.15703378, 0.57917234, 1.1944477 , 1.40219276, 0.70476857,
                              1.23736756, 1.03271783, 0.69775839, 1.22632416, 0.71906011,
                              1.12826509, 1.142113  , 1.22632416, 1.19985586, 0.70585718,
                              0.72652107, 1.22632416, 1.142113  , 1.15703378, 1.22632416,
                              1.19780314, 1.142113  , 1.142113  , 0.72028392, 1.22632416,
                              0.78968558, 0.71906011, 1.19780314, 1.22632416, 1.24462716,
                              0.68456622, 1.23933794, 0.39611022, 0.69585122, 1.15703378,
                              1.19780314, 0.61547879, 1.24462716, 0.7208303 , 1.22632416,
                              0.7018943 , 0.69785836, 1.32591399, 0.82419426, 1.19011568,
                              1.22632416, 1.15703378, 0.69775839, 0.68852529, 1.24462716,
                              1.22632416, 0.70428396, 1.19780314, 0.34753964, 0.71906011,
                              0.84691392, 1.22632416, 1.04749734, 1.22632416, 1.142113  ,
                              0.69775839, 0.72702098, 0.84691392, 1.02224718, 0.72028392]),
                    atol=NUMPY_atol)

    # tree_based_correction_weights with "max_depth = 1"
    assert_allclose(drift_explainer.get_tree_based_correction_weights(max_depth=1),
                    np.array([0.94703727, 0.81154305, 0.80612326, 1.0295544 , 1.0295544 ,
                              0.81154305, 1.20640805, 0.81154305, 1.20640805, 1.0295544 ,
                              1.21451906, 1.0295544 , 0.82080803, 0.82908081, 1.20640805,
                              1.20640805, 0.81154305, 1.20640805, 1.0295544 , 1.0295544 ,
                              0.80612326, 1.0295544 , 1.21451906, 1.20640805, 0.81154305,
                              1.0295544 , 0.81154305, 0.80612326, 1.0295544 , 0.81154305,
                              1.21451906, 1.20640805, 1.20640805, 0.81154305, 1.20640805,
                              1.0295544 , 1.0295544 , 1.20640805, 0.80612326, 0.9700073 ,
                              1.0295544 , 0.83122672, 0.9700073 , 0.94703727, 0.81154305,
                              1.21451906, 0.9700073 , 0.80612326, 1.20640805, 0.81154305,
                              1.0295544 , 1.0295544 , 1.20640805, 0.9700073 , 0.81154305,
                              0.81154305, 1.20640805, 1.0295544 , 1.0295544 , 1.20640805,
                              1.20640805, 1.0295544 , 1.0295544 , 0.81154305, 1.20640805,
                              0.82080803, 0.81154305, 1.20640805, 1.20640805, 1.21451906,
                              0.81154305, 1.21451906, 0.83122672, 0.80612326, 1.0295544 ,
                              1.20640805, 0.84918986, 1.21451906, 0.81154305, 1.20640805,
                              0.81154305, 0.81154305, 0.95784911, 0.82080803, 0.9700073 ,
                              1.20640805, 1.0295544 , 0.80612326, 0.81154305, 1.21451906,
                              1.20640805, 0.81154305, 1.20640805, 0.83122672, 0.81154305,
                              1.0295544 , 1.20640805, 1.0295544 , 1.20640805, 1.0295544 ,
                              0.80612326, 0.81154305, 1.0295544 , 0.96352923, 0.81154305]),
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
    assert drift_explainer.iteration_range == (0, 97)
    assert drift_explainer.n_features == 4
    assert drift_explainer.task == 'classification'
