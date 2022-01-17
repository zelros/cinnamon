import pandas as pd, numpy as np
from numpy.testing import assert_allclose
from sklearn import datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
from cinnamon.drift import ModelDriftExplainer
from cinnamon.drift.drift_utils import (DriftMetricsNum, DriftMetricsCat, assert_drift_metrics_equal,
                                        assert_drift_metrics_list_equal)
from cinnamon.common.stat_utils import BaseStatisticalTestResult, Chi2TestResult


RANDOM_SEED = 2021
NUMPY_atol = 1e-8


def test_boston_xgboost_XGBRegressor():
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
    prediction_drift_ref = [DriftMetricsNum(mean_difference=-0.7907300941865429,
                                            wasserstein=1.0933355933457942,
                                            ks_test=BaseStatisticalTestResult(statistic=0.053783823966696405,
                                                                              pvalue=0.8977000230212033))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(),
                                    prediction_drift_ref)

    # target drift
    assert_drift_metrics_equal(drift_explainer.get_target_drift(),
                               DriftMetricsNum(mean_difference=-0.609240261671129,
                                               wasserstein=1.3178114778471604,
                                               ks_test=BaseStatisticalTestResult(statistic=0.07857567647933393,
                                                                                 pvalue=0.4968030078636394)))

    # performance_metrics_drift
    assert drift_explainer.get_performance_metrics_drift() == {'dataset 1': {'mse': 0.2798235381304802,
                                                                             'explained_variance': 0.9969813455726874},
                                                               'dataset 2': {'mse': 12.459430618659226,
                                                                             'explained_variance': 0.8088751025786054}}

    # tree_based_drift_values "node_size"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='node_size'),
                    np.array([[4.91449084],
                              [0.58104408],
                              [1.27841],
                              [0.06645859],
                              [2.53736065],
                              [6.40104981],
                              [2.77411753],
                              [4.30972915],
                              [0.75707501],
                              [1.76967233],
                              [1.45685282],
                              [2.7046629],
                              [7.52274391]]),
                    atol=NUMPY_atol)

    # tree_based_drift_values "mean"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='mean'),
                    np.array([[0.04875801],
                              [0.01055472],
                              [0.18686353],
                              [-0.00062653],
                              [0.24199658],
                              [-0.54432273],
                              [-0.04957123],
                              [-0.25782332],
                              [0.03362896],
                              [0.0017494],
                              [-0.15995592],
                              [-0.14088634],
                              [-0.03940366]]),
                    atol=NUMPY_atol)

    # tree_based_drift_values "mean_norm"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='mean_norm'),
                    np.array([[0.05357186],
                              [0.00647816],
                              [0.00223468],
                              [-0.00030556],
                              [0.07305206],
                              [-0.30163415],
                              [-0.00561431],
                              [-0.26031085],
                              [0.01544557],
                              [-0.02843446],
                              [-0.01753534],
                              [-0.03112263],
                              [-0.03282566]]),
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
                    np.array([1.17307918, 1.10429956, 1.19169059, 1.13529706, 1.14578311,
                              1.200479, 0.9649062, 1.1040782, 0.72199983, 1.02503113,
                              1.10821682, 0.90692646, 1.14402501, 1.16943841, 0.89823011,
                              0.49860856, 1.05420192, 1.15884044, 0.73639696, 0.82229171,
                              1.21199361, 1.11181694, 1.16128396, 0.89664012, 0.83593215,
                              1.15927253, 1.08385371, 1.19832782, 1.08061481, 0.83970824,
                              0.67884186, 1.08229419, 1.26139195, 0.94657303, 0.98498086,
                              1.03046682, 1.30314848, 0.99200532, 0.93619794, 1.21969226,
                              1.00533594, 1.04201369, 0.95046694, 1.11969526, 0.84742845,
                              1.05551917, 1.16019169, 0.97660996, 1.03919462, 0.98368205,
                              1.16404791, 0.94732921, 0.64433073, 0.92588028, 1.06609024,
                              0.85807047, 1.15019456, 0.9041984, 0.92253366, 1.05986563,
                              0.90813912, 0.79658433, 0.93378748, 0.41866103, 1.11321769,
                              1.09532792, 1.34053538, 0.9097396, 1.05295164, 1.05391176,
                              1.19440804, 1.00559, 1.19822066, 0.54226207, 1.10914477,
                              1.10470429, 1.17477269, 1.00409972, 1.14740161, 0.54482584,
                              0.76272859, 1.03716135, 1.00769731, 0.81684966, 1.04653634,
                              1.17199529, 0.98452679, 0.84458253, 0.7648663, 1.10043305,
                              1.07032943, 0.98022465, 1.11492404, 0.8880634, 0.74218069,
                              1.1932043, 0.95109895, 0.36474178, 1.05405076, 1.06515056,
                              1.06951405, 1.11754207, 0.79994208, 1.11975833, 1.0943148,
                              0.87493754, 0.98749879, 0.93484036, 1.07195407, 1.03911817,
                              1.10384068, 1.22516033, 1.10159258, 0.93925909, 1.06298456,
                              0.80235869, 1.23848544, 1.107232, 0.55608405, 0.73086594,
                              1.05723357, 0.93902368, 0.61808163, 1.02656171, 1.00817501,
                              0.98423577, 1.07234402, 1.09151527, 0.72255239, 0.87157415,
                              1.16842963, 0.92141861, 1.20364012, 1.12475587, 0.86069386,
                              1.07168903, 1.21749038, 0.65690687, 0.90819228, 0.96272015,
                              1.06238916, 1.13443926, 0.844287, 1.15204437, 0.86095568,
                              0.36529403, 0.53725626, 0.85678148, 1.00862856, 0.97270768,
                              1.43748866, 1.10257548, 1.09825674, 1.08947571, 0.37388814,
                              0.53668296, 1.23500601, 1.12470947, 1.19278362, 0.94510303,
                              1.00776462, 1.20820062, 1.06303224, 1.05105978, 1.29058967,
                              0.63277066, 0.60221109, 1.16910423, 1.16981277, 1.11942871,
                              0.9537934, 0.98443738, 1.0849316, 0.8290222, 1.00789384,
                              0.61950957, 1.02245451, 0.8330702, 1.04320452, 0.96021991,
                              0.80166563, 0.99422759, 1.03418793, 1.14262089, 0.86111115,
                              1.06633001, 1.10596964, 1.1413938, 0.96951136, 1.08869194,
                              1.11767264, 1.19460782, 1.24016088, 1.03017977, 1.03639227,
                              1.04536011, 1.04680402, 1.09302331, 0.95333154, 1.26187366,
                              0.85062876, 1.05865228, 1.10202065, 0.76674624, 0.90350575,
                              1.33108946, 0.46632369, 1.21149765, 0.77911796, 1.03232501,
                              1.16269854, 1.07684434, 0.85179834, 1.13179073, 0.96137986,
                              0.96767259, 0.93977526, 0.94300641, 0.96423001, 1.04319695,
                              0.75998146, 0.81221585, 0.97268292, 0.90971505, 0.87193452,
                              1.13512028, 1.06957513, 1.12052159, 1.05101438, 1.07174728,
                              1.14397345, 0.96879791, 1.00484637, 1.020123, 1.11576274,
                              1.27949005, 1.00969895, 1.23093211, 0.87762165, 1.11754222,
                              1.06167064, 1.11207849, 1.03891518, 1.04050737, 1.31171655,
                              1.19664899, 1.14727933, 0.95127623, 0.4850944, 0.97516067,
                              0.61839221, 1.0923905, 0.97710356, 0.90319709, 0.37564845,
                              0.98990542, 1.31876529, 1.0951119, 1.10194228, 0.66620351,
                              0.93853275, 1.1338095, 0.83854722, 1.09114115, 0.88133065,
                              0.7946076, 1.14922025, 0.81525115, 1.11380585, 1.07034183,
                              0.79821389, 0.95217574, 0.95778946, 0.53445616, 0.78168331,
                              1.13640662, 0.90417136, 1.00300019, 1.0666356, 1.06334913,
                              1.10286597, 1.07458109, 1.11791495, 1.28759224, 1.12574537,
                              1.0835987, 1.0043716, 1.13099684, 1.04718128, 0.92558712,
                              0.96849863, 1.1118371, 1.14082817, 0.98227235, 0.91048265,
                              1.23001552, 0.97053114, 0.68878288, 1.01220493, 1.00796822,
                              0.91582841, 1.04847328, 0.95905179, 0.87059411, 1.11877684,
                              1.19680597, 1.08497233, 0.97513341, 0.94617059, 1.16122938,
                              1.1232211, 0.7546364, 1.10187586, 1.01408795, 1.02770676,
                              1.11252479, 1.11156682, 1.07147014, 0.92417427, 1.10953318,
                              1.00213953, 1.04189862, 0.6621549, 1.09260932, 1.10529048,
                              0.87546568, 1.11195891, 0.77509501, 1.00462888, 0.83154827,
                              1.15078635, 0.85786918, 0.79800891, 1.11380145, 1.04628851,
                              1.1658299, 1.08542816, 0.92665663, 0.92427178, 1.08199608,
                              0.77310855, 1.06173834, 1.14117264, 1.08938491, 1.19071799,
                              1.1036898, 1.17283324, 1.10692978, 1.10861571, 1.14965191,
                              0.99824981, 1.02571374, 1.22055746, 1.09707912]),
                    atol=NUMPY_atol)

    # tree_based_correction_weights with "max_depth = 2"
    assert_allclose(drift_explainer.get_tree_based_correction_weights(max_depth=2),
                    np.array([1.0582569, 1.00670071, 1.02780781, 1.01150329, 1.03688912,
                              1.00576777, 1.00612103, 1.01511992, 1.01322022, 1.03347478,
                              1.02954666, 0.99914983, 1.05850012, 1.0292191, 0.97294076,
                              0.82615829, 1.02780781, 1.05521708, 0.99188371, 0.9831289,
                              1.01166509, 0.90378646, 1.05370445, 1.01967054, 1.0353219,
                              1.0232752, 1.01800961, 1.0273559, 0.98887101, 0.95886379,
                              1.04501309, 1.0525534, 1.05312083, 1.02271887, 1.05104311,
                              1.00471757, 1.0343851, 1.01738836, 1.01233581, 1.061134,
                              1.04299697, 1.01421838, 1.0228959, 1.04469885, 0.9894954,
                              1.00265687, 1.03623705, 1.03631859, 1.01358135, 1.01597777,
                              1.02172966, 1.02204174, 0.83166848, 0.93117947, 1.01605008,
                              1.01469444, 0.91661529, 1.01967054, 1.03592465, 1.01741396,
                              1.00382789, 1.04469193, 1.00141397, 0.74728645, 1.02906745,
                              1.06031501, 1.05473691, 1.02666007, 1.00488201, 1.00933517,
                              1.00658136, 1.04340723, 1.02903632, 0.82524041, 1.0172848,
                              1.02000109, 1.02995608, 0.8681523, 1.03138448, 0.79215619,
                              1.04684971, 1.01526221, 1.00804983, 0.79556459, 1.0068124,
                              1.02930453, 1.04688084, 1.00587795, 0.78993104, 1.00870625,
                              0.96219822, 1.01430715, 1.04469193, 0.91657232, 0.83016682,
                              1.02802593, 1.04568853, 0.73346207, 1.02750187, 1.00962653,
                              1.02027967, 1.02206637, 1.0353219, 1.03623705, 1.03187405,
                              0.98157129, 1.00775198, 1.01138531, 1.01465086, 1.02172783,
                              1.04511281, 1.0243181, 1.02986528, 1.03315717, 1.02750187,
                              1.00290184, 1.01612785, 1.05724868, 0.82524041, 0.78535912,
                              1.00020967, 0.87923036, 0.8396892, 1.01402176, 1.01597813,
                              1.00230231, 1.01417279, 1.02202794, 0.84071042, 1.01760687,
                              1.02494963, 1.05797844, 1.03436595, 1.01038303, 1.05506159,
                              1.0196529, 1.03639186, 0.93657553, 0.88233607, 1.02506785,
                              1.02436659, 1.05132528, 1.0019079, 1.03772613, 1.00775198,
                              0.64759301, 0.78293874, 1.02119678, 1.01898273, 1.01732744,
                              1.05724568, 1.02266183, 1.0252554, 1.02009781, 0.63912218,
                              0.83505949, 1.04158229, 1.01954571, 1.02081984, 1.03386576,
                              0.87173121, 1.05248492, 1.01818577, 1.02021987, 1.05252078,
                              0.72940923, 0.86101295, 1.0310369, 1.02842619, 1.05071831,
                              1.01300651, 1.02150085, 1.01597777, 1.02544089, 1.00910509,
                              0.79240637, 1.01855657, 0.99872639, 1.02874935, 1.04683248,
                              1.04688084, 0.98525423, 1.01169174, 1.02867249, 1.00089645,
                              1.07349237, 1.00822084, 1.07375274, 1.03315717, 1.01334828,
                              1.03108635, 1.0176312, 1.06534155, 1.03928143, 1.01845684,
                              1.02264917, 1.05232278, 1.02688857, 1.02371771, 1.05921163,
                              1.00610203, 1.05903805, 1.02161461, 0.85472849, 1.00022545,
                              1.21358172, 0.74728645, 1.03217915, 0.78535912, 1.04688084,
                              1.04949284, 1.03315717, 0.94982671, 1.03436595, 1.02215344,
                              1.02366864, 1.02688472, 0.96627752, 1.02045449, 1.01117574,
                              1.0353219, 1.03528255, 1.01741396, 1.01439825, 1.03131391,
                              1.04822021, 1.02874935, 1.01227542, 1.02888112, 1.01233581,
                              1.01078483, 0.86769516, 1.02550814, 1.01684283, 1.0338169,
                              1.04651762, 1.01388886, 1.06227251, 1.0198984, 1.01353115,
                              1.02565277, 1.02964456, 1.01227542, 1.02665291, 1.03480824,
                              1.04360213, 1.01016196, 1.00640105, 0.94432931, 1.0171349,
                              0.87698451, 1.01815217, 1.03185809, 1.04755748, 0.73247884,
                              1.03704095, 1.05824615, 1.0358401, 1.00699825, 0.80067744,
                              1.03347478, 1.03563177, 1.03315717, 1.0405324, 1.01384475,
                              1.02888112, 1.02067834, 1.01759046, 1.01619322, 1.01848426,
                              1.02565277, 0.87067231, 1.00203294, 0.7774344, 1.00711629,
                              1.03649701, 1.01469444, 1.00670071, 1.02917602, 1.02416993,
                              1.00986349, 1.0257819, 1.0248167, 1.03356106, 1.02874553,
                              1.04224102, 1.04211948, 1.02562278, 1.05082067, 1.03315717,
                              1.0353219, 1.01435257, 1.01429057, 0.90506559, 1.04282809,
                              1.03309092, 1.01117834, 0.94065906, 1.01738094, 1.01159049,
                              1.02762745, 1.01227542, 1.0196529, 1.00291289, 1.03501383,
                              1.02626132, 1.01001231, 1.00710689, 1.033865, 1.02297717,
                              1.00776766, 1.03347478, 1.02917602, 1.00804983, 0.9844154,
                              0.97253672, 1.0358401, 1.00986349, 1.00235212, 1.02034128,
                              1.01066513, 1.03388299, 1.01900233, 1.04207765, 1.03390331,
                              0.9927288, 0.92165943, 0.98165689, 0.9094749, 1.04221711,
                              1.02917602, 1.00775198, 0.89860475, 1.02330345, 1.03180699,
                              1.04223021, 1.01741943, 0.99429937, 1.0069181, 1.03704095,
                              1.00775198, 1.02750339, 1.01588284, 0.90758758, 1.0465381,
                              1.01198529, 1.02954706, 1.02780781, 1.01883045, 1.0258135,
                              1.03223902, 0.88213941, 1.06822964, 1.00670071]),
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
    assert drift_explainer.iteration_range == (0, 163)
    assert drift_explainer.n_features == 13
    assert drift_explainer.task == 'regression'


def test_breast_cancer_xgboost_XGBClassifier():
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
    prediction_drift_ref = [DriftMetricsNum(mean_difference=0.0018858597634534568,
                                            wasserstein=0.40122835384598343,
                                            ks_test=BaseStatisticalTestResult(statistic=0.08661729701137265,
                                                                              pvalue=0.30851650924557306))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(),
                                    prediction_drift_ref)

    # prediction drift "proba"
    prediction_drift_proba_ref = [DriftMetricsNum(mean_difference=0.00845238602309173,
                                                  wasserstein=0.024145883189508617,
                                                  ks_test=BaseStatisticalTestResult(statistic=0.08661729701137265,
                                                                                    pvalue=0.30851650924557306))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(prediction_type='proba'),
                                    prediction_drift_proba_ref)

    # target drift
    assert_drift_metrics_equal(drift_explainer.get_target_drift(),
                               DriftMetricsCat(wasserstein=0.0024097093655411628,
                                               chi2_test=Chi2TestResult(statistic=0.0,
                                                                        pvalue=1.0,
                                                                        dof=1,
                                                                        contingency_table=pd.DataFrame([[148.0, 250.0], [64.0, 107.0]],
                                                                                                       index=['X1', 'X2'], columns=[0, 1]))))

    # performance_metrics_drift
    assert drift_explainer.get_performance_metrics_drift() == {'dataset 1': {'log_loss': 0.013343524769294877},
                                                               'dataset 2': {'log_loss': 0.11184813333838656}}


    # tree_based_drift_values "node_size"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='node_size'),
                    np.array([[0.1263785 ],
                              [1.27306046],
                              [0.03015075],
                              [0.63991709],
                              [0.48471829],
                              [0.01076813],
                              [0.52800904],
                              [0.62091044],
                              [0.00502513],
                              [0.02160804],
                              [0.05284761],
                              [0.16747726],
                              [0.01615219],
                              [0.24027569],
                              [0.06186914],
                              [0.63923858],
                              [0.13051384],
                              [0.00904523],
                              [0.26556225],
                              [0.06645043],
                              [0.970708  ],
                              [0.83135717],
                              [1.1970857 ],
                              [1.27954307],
                              [1.8764872 ],
                              [0.09348557],
                              [0.6073105 ],
                              [1.5896039 ],
                              [0.39138964],
                              [0.05313921]]),
                    atol=NUMPY_atol)

    # tree_based_drift_values "mean"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='mean'),
                    np.array([[-9.33158904e-03],
                              [ 1.08491251e-02],
                              [ 2.25198024e-03],
                              [-8.03116350e-03],
                              [ 4.24347320e-03],
                              [ 1.24361599e-03],
                              [ 2.26169711e-02],
                              [-2.12891656e-03],
                              [ 9.47460649e-06],
                              [ 4.13937125e-03],
                              [ 1.80148733e-03],
                              [-1.81682108e-03],
                              [ 7.62334093e-05],
                              [ 1.01233202e-03],
                              [-2.89777474e-03],
                              [-1.29647806e-02],
                              [ 4.80747869e-03],
                              [ 1.99959485e-03],
                              [ 2.11957969e-03],
                              [ 1.51197027e-03],
                              [ 2.62156736e-03],
                              [-1.34094254e-03],
                              [-2.26530507e-02],
                              [-3.61079870e-03],
                              [ 1.80279903e-02],
                              [-5.65097244e-03],
                              [ 1.42406917e-03],
                              [-7.61147393e-03],
                              [ 1.94904924e-03],
                              [-2.78114137e-03]]),
                    atol=NUMPY_atol)

    # tree_based_drift_values "mean_norm"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='mean_norm'),
                    np.array([[ 0.00128017],
                              [-0.00882363],
                              [ 0.00227804],
                              [-0.00959588],
                              [ 0.01738345],
                              [ 0.00044783],
                              [ 0.01318167],
                              [ 0.00522705],
                              [-0.0001583 ],
                              [-0.00016892],
                              [ 0.00055728],
                              [ 0.00208376],
                              [-0.00037917],
                              [-0.00242911],
                              [-0.00055455],
                              [-0.01183045],
                              [ 0.00458787],
                              [-0.00033537],
                              [ 0.00566654],
                              [-0.00038043],
                              [-0.00204518],
                              [ 0.00529169],
                              [-0.02016067],
                              [-0.02651999],
                              [ 0.02481481],
                              [-0.00331497],
                              [ 0.01063788],
                              [-0.00117099],
                              [ 0.00806287],
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
                    np.array([1.08428656, 0.98779765, 1.01004425, 1.01198344, 1.02879822,
                              0.98773117, 1.04529611, 0.85822284, 0.91015066, 0.96450881,
                              1.42437081, 1.00798061, 0.98089596, 1.06205024, 0.97622795,
                              1.00012326, 0.98645635, 1.0261558 , 1.22949513, 0.9865238 ,
                              1.03152298, 0.98019005, 0.98338215, 0.97560413, 1.17480894,
                              0.95448352, 0.97939546, 0.98827448, 0.98063303, 0.98074952,
                              1.01734749, 0.94960169, 0.96228268, 0.97982364, 0.91324907,
                              1.04500548, 0.91749073, 0.98773117, 0.88479926, 1.08202122,
                              1.00741486, 0.98063303, 0.9469353 , 0.99688845, 0.89881648,
                              0.98060956, 1.03509149, 0.97042311, 1.06430047, 0.9835708 ,
                              1.00281669, 1.18462719, 1.00798061, 0.99350168, 0.96513333,
                              0.9739485 , 0.97494372, 0.98779765, 1.01164044, 0.98642777,
                              0.99597257, 0.97272405, 0.98157408, 0.9059657 , 0.9301798 ,
                              0.96524431, 0.93370664, 0.99740793, 1.10960388, 0.98292363,
                              0.94462827, 0.98779765, 0.9670841 , 1.04846226, 0.93114761,
                              0.98942195, 0.96816803, 1.03360233, 0.933209  , 0.91749073,
                              1.11322141, 1.03708278, 1.08825504, 0.98679061, 0.99132777,
                              0.94757054, 0.90240473, 0.90240473, 1.05627803, 0.98304765,
                              1.15088249, 1.0114699 , 0.90265483, 1.01727811, 0.8852331 ,
                              1.06640841, 1.12758798, 1.00624029, 0.97626879, 1.0919997 ,
                              0.98384339, 0.99027048, 0.94307742, 1.13382257, 0.91749073,
                              0.98371292, 0.97158533, 0.93499571, 0.98063303, 1.00906631,
                              1.17390193, 1.00350997, 1.00798061, 1.00774618, 0.93337891,
                              0.91749073, 0.98089596, 1.05158508, 1.01198344, 0.98773117,
                              0.9162551 , 0.88907695, 1.00350997, 0.98773117, 1.36914166,
                              0.93871747, 0.93483678, 0.92323832, 0.94440601, 0.97003715,
                              1.05828061, 0.97371772, 0.91634223, 1.05073273, 1.03353156,
                              0.98164014, 0.97478932, 1.10060063, 0.92091737, 0.99264768,
                              1.12282325, 0.88636545, 0.966354  , 1.10481234, 0.99869285,
                              0.90988848, 0.97931488, 0.96300827, 1.04216947, 1.06545006,
                              0.98355065, 0.90147347, 1.03535353, 1.06927459, 0.93119277,
                              1.0280631 , 0.9180559 , 0.89861645, 1.12900094, 0.9275707 ,
                              1.18907919, 1.06734279, 0.98754364, 0.9865238 , 1.16398779,
                              0.9244803 , 1.02336605, 0.97003715, 0.96849676, 0.93291755,
                              0.98773117, 1.01750398, 0.98145828, 0.97609946, 0.90469468,
                              1.0074691 , 1.13180722, 1.00139293, 0.87502637, 0.93263014,
                              0.96892833, 0.96399035, 0.99714423, 0.96674607, 1.06716869,
                              0.9919385 , 0.89895245, 0.93216506, 0.97271992, 0.9876812 ,
                              0.98773117, 0.9614012 , 0.96086446, 0.98773117, 0.98074952,
                              1.0127484 , 1.01183364, 1.06928446, 1.2215748 , 0.97467699,
                              0.98355065, 0.97596621, 1.04981039, 1.02485891, 0.97042311,
                              1.02524323, 1.27606744, 1.00945596, 1.10246661, 1.17136507,
                              0.97643855, 0.98147592, 0.9782614 , 0.89881648, 1.03323055,
                              1.08108668, 0.91090016, 0.98063303, 0.98138083, 0.91324907,
                              1.07469582, 0.94141091, 1.06280235, 0.93047142, 0.86903393,
                              0.97560413, 0.9978341 , 0.9607891 , 0.98776875, 0.99012808,
                              1.13098607, 1.00332061, 0.92909198, 0.9865238 , 1.00250421,
                              1.0250775 , 1.02638397, 1.00671884, 1.0114699 , 1.19838307,
                              0.98726919, 1.10437427, 0.98039592, 0.94427415, 1.18654083,
                              0.92901024, 1.13773012, 0.99256936, 0.94396672, 0.99084782,
                              1.14691129, 0.97931488, 1.04075873, 0.90012064, 0.98773117,
                              1.14410936, 0.95658562, 0.95646865, 0.96802085, 1.0475791 ,
                              1.01285327, 0.97609946, 0.97860627, 1.03986432, 0.94588329,
                              1.20479445, 0.92841088, 0.91924734, 1.00521415, 0.97012606,
                              0.97640394, 1.0114699 , 0.96653947, 0.9881296 , 1.00332061,
                              0.97232891, 1.02752766, 0.99469504, 1.04675585, 0.99449748,
                              0.92489701, 1.17523107, 1.01047398, 0.92147771, 1.01100466,
                              1.04404874, 0.96465678, 0.89823282, 0.93055917, 0.94199267,
                              1.02234729, 0.97857676, 0.9801354 , 1.00009178, 0.98089596,
                              1.02960008, 0.9733294 , 0.99466545, 0.91574895, 1.03887735,
                              1.00115266, 0.9870488 , 0.98141858, 0.92590087, 0.99161812,
                              0.97534307, 1.03049288, 0.97609946, 0.98773117, 1.049548  ,
                              0.87772218, 1.00650032, 0.95211959, 0.91095038, 0.9944623 ,
                              1.01672036, 1.03942196, 0.94307742, 0.98773117, 1.00671884,
                              1.00406196, 0.99987447, 0.91690051, 0.98063303, 1.13836702,
                              0.99609838, 0.97075206, 0.97305664, 0.98122128, 1.09539349,
                              1.04053165, 1.01393336, 0.96315339, 0.99538795, 0.97948131,
                              0.8968078 , 0.84907082, 1.04501469, 0.91239853, 0.98007274,
                              0.97271965, 1.1463738 , 1.04029276, 0.96206865, 0.98488055,
                              0.97006857, 1.12813169, 1.00798061, 1.36463887, 1.06215117,
                              1.20938883, 0.9870488 , 0.98955059, 0.91515789, 0.94427038,
                              1.02741124, 0.98063303, 0.98157408, 0.99352529, 1.05546537,
                              0.98955059, 0.92840611, 1.05289134, 1.00835718, 0.96408195,
                              1.01112385, 0.98378468, 1.09005511, 0.89483643, 0.99248536,
                              0.98355065, 0.88629458, 0.97878493, 0.90006216, 1.00042949,
                              0.84904005, 0.97609946, 1.0842126 , 0.87290672, 0.8982086 ,
                              1.09905586, 1.03169785, 0.97155954, 1.12331844, 1.03296381,
                              1.03147566, 1.02338469, 0.89823282, 1.08403779, 1.05719262,
                              0.96004978, 1.0410868 , 1.05459072, 1.02844853, 1.06039393,
                              0.99858696, 0.96570366, 0.99264768]),
                    atol=NUMPY_atol)

    # tree_based_correction_weights with "max_depth = 1"
    assert_allclose(drift_explainer.get_tree_based_correction_weights(max_depth=1),
                    np.array([0.98280665, 1.01395788, 1.01741124, 0.98457398, 0.99753022,
                              1.01395788, 1.0152764 , 0.96703698, 0.99568282, 0.98457398,
                              1.04677731, 0.98457398, 1.01395788, 1.02673165, 1.00264515,
                              1.01395788, 0.98457398, 0.99568282, 1.03542404, 1.01395788,
                              0.99568282, 0.98457398, 1.01395788, 0.98457398, 1.01882596,
                              1.00605998, 1.00264515, 1.01395788, 0.98457398, 0.97636052,
                              0.98457398, 0.99777194, 1.00264515, 0.98457398, 0.98457398,
                              1.01395788, 0.98457398, 1.01395788, 0.98623511, 1.01751935,
                              0.99753022, 0.98457398, 0.99865651, 0.98457398, 0.97636052,
                              0.97523168, 1.01395788, 1.00264515, 1.0152764 , 1.01395788,
                              1.01395788, 1.03042038, 0.98457398, 0.99766728, 0.97636052,
                              1.00264515, 1.00264515, 1.01395788, 1.00264515, 0.98457398,
                              1.01409326, 1.00264515, 1.01395788, 0.97636052, 0.97636052,
                              0.98457398, 1.01561189, 0.99568282, 1.01284915, 1.01513155,
                              0.99865651, 1.01395788, 0.99753022, 1.0152764 , 1.00605998,
                              0.97523168, 1.00264515, 1.0152764 , 0.98457398, 0.98457398,
                              1.01097338, 1.01395788, 1.02489902, 0.98457398, 0.96593755,
                              1.00050943, 0.98457398, 0.98457398, 1.00181215, 1.01395788,
                              1.04338737, 0.98457398, 0.98194479, 1.01395788, 0.98737669,
                              1.0152764 , 1.02538402, 1.01395788, 0.98457398, 1.0096021 ,
                              1.00531858, 1.01395788, 0.98457398, 1.02673165, 0.98457398,
                              0.97636052, 0.99753022, 0.97636052, 0.98457398, 1.00605998,
                              1.02251836, 1.01395788, 0.98457398, 0.98457398, 0.99993769,
                              0.98457398, 1.01395788, 1.00579625, 0.98457398, 1.01395788,
                              0.98457398, 0.98457398, 1.01395788, 1.01395788, 1.03032126,
                              0.98457398, 0.96482076, 0.99777194, 0.98457398, 1.00264515,
                              1.01097338, 1.00605998, 0.98457398, 0.99504599, 1.0152764 ,
                              1.01395788, 1.00264515, 1.02251836, 0.98737669, 0.98457398,
                              1.0207163 , 0.96399579, 1.00264515, 1.02673165, 1.02580764,
                              0.99568282, 0.98457398, 1.01208006, 1.0152764 , 1.0152764 ,
                              0.98457398, 0.97636052, 1.01395788, 1.03284544, 1.01864211,
                              1.01395788, 0.99568282, 0.98640077, 1.01912013, 0.98457398,
                              1.03343599, 1.0152764 , 0.99766728, 1.01395788, 1.01492   ,
                              0.98847042, 1.01395788, 1.00264515, 1.00264515, 0.98457398,
                              1.01395788, 0.98457398, 1.00264515, 0.98457398, 0.98457398,
                              0.98457398, 1.02673165, 0.99568282, 0.97636052, 0.99865651,
                              1.00453482, 1.00264515, 1.0152764 , 1.00264515, 1.01297731,
                              0.98457398, 0.98457398, 0.98623511, 0.98457398, 0.98457398,
                              1.01395788, 1.00264515, 1.00341664, 1.01395788, 0.97636052,
                              1.01395788, 0.98457398, 1.0152764 , 1.02983798, 1.00264515,
                              0.98457398, 1.00078827, 1.00341664, 1.02163105, 1.00264515,
                              0.9999824 , 1.02834176, 0.99590089, 1.01940727, 1.02780077,
                              1.00264515, 0.98457398, 1.01395788, 0.97636052, 1.0019467 ,
                              1.0152764 , 0.98934674, 0.98457398, 0.98457398, 0.98457398,
                              1.0152764 , 1.00078827, 1.0152764 , 0.99673785, 0.97522273,
                              0.98457398, 1.00605998, 0.98737669, 0.99568282, 0.98457398,
                              0.99901535, 0.98457398, 0.99914048, 1.01395788, 0.98457398,
                              1.01741124, 1.03061875, 0.98457398, 0.98457398, 1.01923606,
                              0.98457398, 0.9875145 , 1.0138744 , 0.98865755, 1.02489902,
                              0.98457398, 1.01882596, 0.98457398, 1.00078827, 0.98457398,
                              1.03223169, 0.98457398, 0.98737669, 0.9875145 , 1.01395788,
                              1.03223293, 0.99568282, 1.00799942, 1.00050943, 1.01741124,
                              0.98457398, 0.98457398, 0.99962322, 1.01395788, 0.98457398,
                              1.01794186, 0.97451715, 0.99568282, 1.01395788, 1.00264515,
                              0.99568282, 0.98457398, 0.98640077, 1.00680682, 0.98457398,
                              0.98640077, 0.99568282, 0.98457398, 1.01741124, 0.99766728,
                              0.98457398, 1.03409739, 1.00264515, 0.98457398, 1.01395788,
                              0.99568282, 1.00264515, 0.98457398, 0.99777194, 1.00078827,
                              1.01395788, 1.00050943, 0.98457398, 1.01395788, 1.01395788,
                              1.00911906, 0.97636052, 1.00264515, 0.98561724, 0.99568282,
                              0.98457398, 1.01395788, 0.98457398, 0.99865651, 0.97523168,
                              1.01208006, 1.00002232, 0.98457398, 1.01395788, 1.0152764 ,
                              1.0181665 , 0.98457398, 0.99766728, 0.97636052, 0.99568282,
                              1.00680682, 0.98280665, 0.98457398, 1.01395788, 0.98457398,
                              1.01395788, 0.98457398, 0.98737669, 0.98457398, 1.03211064,
                              1.01239845, 1.00605998, 1.00264515, 0.98457398, 1.02136956,
                              1.0152764 , 1.01395788, 1.00078827, 0.98457398, 0.98525128,
                              0.97636052, 0.97636052, 1.0152764 , 0.97451715, 0.98457398,
                              0.98457398, 1.01923606, 1.00605998, 0.97503619, 0.99568282,
                              0.98457398, 1.02673165, 0.98457398, 1.0361225 , 1.01751935,
                              1.03072797, 1.01395788, 1.01395788, 0.98457398, 1.00166753,
                              0.99568282, 0.98457398, 1.01395788, 0.98457398, 1.02316241,
                              1.01395788, 0.98457398, 1.01462704, 1.01395788, 1.00264515,
                              0.97636052, 1.01208006, 1.01751935, 0.99568282, 1.01395788,
                              0.98457398, 0.97636052, 0.99568282, 0.98561724, 1.00605998,
                              0.97451715, 0.98457398, 1.0152764 , 0.98457398, 0.98279763,
                              1.01751935, 1.01395788, 1.00296057, 1.02673165, 1.01395788,
                              0.99568282, 1.01395788, 0.98457398, 0.99755085, 0.95547432,
                              0.98457398, 1.01395788, 1.0152764 , 1.00834612, 1.01460807,
                              1.00050943, 0.97636052, 0.98457398]),
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
    assert drift_explainer.iteration_range == (0, 146)
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
    prediction_drift_ref = [DriftMetricsNum(mean_difference=0.33904417620764843, wasserstein=0.3390441762076484, ks_test=BaseStatisticalTestResult(statistic=0.06349206349206349, pvalue=0.9987212484986797)),
                            DriftMetricsNum(mean_difference=0.3479284484826383, wasserstein=0.3566064995077869, ks_test=BaseStatisticalTestResult(statistic=0.1365079365079365, pvalue=0.5571746191565534)),
                            DriftMetricsNum(mean_difference=-0.6179708909184214, wasserstein=0.6183046163784134, ks_test=BaseStatisticalTestResult(statistic=0.17142857142857143, pvalue=0.2821678346768163))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(),
                                    prediction_drift_ref)

    # prediction drift "proba"
    prediction_drift_proba_ref = [DriftMetricsNum(mean_difference=0.06179329878133205, wasserstein=0.06183055994795665, ks_test=BaseStatisticalTestResult(statistic=0.1111111111111111, pvalue=0.793799989988573)),
                                  DriftMetricsNum(mean_difference=0.08402968007065947, wasserstein=0.0844271551403735, ks_test=BaseStatisticalTestResult(statistic=0.10793650793650794, pvalue=0.8205934119780005)),
                                  DriftMetricsNum(mean_difference=-0.1458229802755846, wasserstein=0.14582381487365756, ks_test=BaseStatisticalTestResult(statistic=0.17142857142857143, pvalue=0.2821678346768163))]
    assert_drift_metrics_list_equal(drift_explainer.get_prediction_drift(prediction_type='proba'),
                                    prediction_drift_proba_ref)

    # target drift
    assert_drift_metrics_equal(drift_explainer.get_target_drift(),
                               DriftMetricsCat(wasserstein=0.09523809523809523,
                                               chi2_test=Chi2TestResult(statistic=1.3333333333333333,
                                                                        pvalue=0.5134171190325922,
                                                                        dof=2,
                                                                        contingency_table=pd.DataFrame([[33.0, 34.0, 38.0], [17.0, 16.0, 12.0]],
                                                                                                       index=['X1', 'X2'], columns=[0, 1, 2]))))

    # performance_metrics_drift
    assert drift_explainer.get_performance_metrics_drift() == {'dataset 1': {'log_loss': 0.03573701255733058},
                                                               'dataset 2': {'log_loss': 0.1726300963304109}}

    # tree_based_drift_values "node_size"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='node_size'),
                    np.array([[ 2.24101843],
                              [ 4.95002698],
                              [18.68081445],
                              [29.91475275]]),
                    atol=NUMPY_atol)

    # tree_based_drift_values "mean"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='mean'),
                    np.array([[ 0.        ,  0.18602155, -0.02790859],
                              [ 0.        , -0.00947687, -0.06789058],
                              [ 0.33904412,  0.03495713, -0.26869759],
                              [ 0.        ,  0.13642661, -0.25347424]]),
                    atol=NUMPY_atol)

    # tree_based_drift_values "mean_norm"
    assert_allclose(drift_explainer.get_tree_based_drift_values(type='mean_norm'),
                    np.array([[ 0.        ,  0.02024519,  0.00498964],
                              [ 0.        , -0.01272634,  0.02345512],
                              [ 0.33904412, -0.11905975,  0.0417556 ],
                              [ 0.        ,  0.08622845, -0.1998876 ]]),
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
                    np.array([1.51963273, 0.69958012, 0.68959361, 1.14262557, 1.14070412,
                              0.75359899, 1.22202441, 0.71032805, 1.22202441, 1.13668373,
                              1.23114095, 1.13668373, 0.58041605, 0.9425652 , 1.22202441,
                              1.22202441, 0.71073325, 1.22202441, 1.14242865, 1.14070412,
                              0.69080126, 1.06257767, 1.23114095, 1.22202441, 0.70189534,
                              1.14495092, 0.71073325, 0.69080126, 1.13668373, 0.7200171 ,
                              1.23114095, 1.22202441, 1.22202441, 0.69088332, 1.19747109,
                              1.15991488, 1.14242865, 1.22202441, 0.68959361, 1.05799033,
                              1.16698598, 0.55307216, 1.18901321, 1.4081631 , 0.70229573,
                              1.23114095, 1.05799033, 0.69080126, 1.22202441, 0.70928591,
                              1.14376723, 1.14493691, 1.22202441, 1.20825593, 0.69744213,
                              0.71538202, 1.22202441, 1.14493691, 1.16840963, 1.22202441,
                              1.19747109, 1.13840217, 1.13668373, 0.71028659, 1.22202441,
                              0.77157863, 0.70928591, 1.19747109, 1.22202441, 1.23767279,
                              0.67995558, 1.24226541, 0.40363958, 0.68959361, 1.16412789,
                              1.19747109, 0.62371472, 1.23767279, 0.71073325, 1.22202441,
                              0.69419428, 0.69088332, 1.34032219, 0.79529356, 1.20011855,
                              1.22202441, 1.16840963, 0.69080126, 0.68321417, 1.23767279,
                              1.22202441, 0.70189534, 1.19747109, 0.36267417, 0.70928591,
                              0.89733431, 1.22202441, 1.071889  , 1.22202441, 1.13840217,
                              0.69080126, 0.7157901 , 0.89733431, 1.04908937, 0.71028659]),
                    atol=NUMPY_atol)

    # tree_based_correction_weights with "max_depth = 1"
    assert_allclose(drift_explainer.get_tree_based_correction_weights(max_depth=1),
                    np.array([0.93952122, 0.82672536, 0.82205507, 1.02474169, 1.02474169,
                              0.84271961, 1.18403958, 0.82663457, 1.18403958, 1.02474169,
                              1.19063563, 1.02474169, 0.83510573, 0.84141768, 1.18403958,
                              1.18403958, 0.82663457, 1.18403958, 1.02474169, 1.02474169,
                              0.82205507, 1.04468159, 1.19063563, 1.18403958, 0.82672536,
                              1.0482974 , 0.82663457, 0.82205507, 1.02474169, 0.82663457,
                              1.19063563, 1.18403958, 1.18403958, 0.82663457, 1.18403958,
                              1.04456687, 1.02474169, 1.18403958, 0.82205507, 0.97290171,
                              1.02462916, 0.85600429, 0.97300855, 0.95780286, 0.82672536,
                              1.19063563, 0.97290171, 0.82205507, 1.18403958, 0.82663457,
                              1.02462916, 1.04468159, 1.18403958, 0.97290171, 0.82663457,
                              0.82663457, 1.18403958, 1.04468159, 1.02462916, 1.18403958,
                              1.18403958, 1.02474169, 1.02474169, 0.82663457, 1.18403958,
                              0.83519744, 0.82663457, 1.18403958, 1.18403958, 1.19063563,
                              0.82663457, 1.21380356, 0.85600429, 0.82205507, 1.02462916,
                              1.18403958, 0.88836458, 1.19063563, 0.82663457, 1.18403958,
                              0.82663457, 0.82663457, 0.94914921, 0.83510573, 0.97290171,
                              1.18403958, 1.02462916, 0.82205507, 0.82663457, 1.19063563,
                              1.18403958, 0.82672536, 1.18403958, 0.85609829, 0.82663457,
                              1.0482974 , 1.18403958, 1.04456687, 1.18403958, 1.02474169,
                              0.82205507, 0.82663457, 1.0482974 , 0.96751189, 0.82663457]),
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
    assert drift_explainer.iteration_range == (0, 117)
    assert drift_explainer.n_features == 4
    assert drift_explainer.task == 'classification'
