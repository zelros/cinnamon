import numpy as np
import scipy.special
import json
import struct
from distutils.version import LooseVersion
from .utils import assert_import, record_import_error, safe_isinstance
from .single_tree import SingleTree, IsoTree
import warnings

warnings.formatwarning = lambda msg, *args, **kwargs: str(msg) + '\n' # ignore everything except the message

# pylint: disable=unsubscriptable-object

try:
    from .. import _cext
except ImportError as e:
    record_import_error("cext", "C extension was not built during install!", e)

try:
    import pyspark
except ImportError as e:
    record_import_error("pyspark", "PySpark could not be imported!", e)

output_transform_codes = {
    "identity": 0,
    "logistic": 1,
    "logistic_nlogloss": 2,
    "squared_loss": 3
}

feature_perturbation_codes = {
    "interventional": 0,
    "tree_path_dependent": 1,
    "global_path_dependent": 2
}


class TreeEnsemble:
    """ An ensemble of decision trees.

    This object provides a common interface to many different types of models.
    """

    def __init__(self, model, data=None, data_missing=None, model_output=None):
        self.model_type = "internal"
        self.trees = None
        self.base_offset = 0
        self.model_output = model_output
        self.objective = None # what we explain when explaining the loss of the model
        self.tree_output = None # what are the units of the values in the leaves of the trees
        self.internal_dtype = np.float64
        self.input_dtype = np.float64 # for sklearn we need to use np.float32 to always get exact matches to their predictions
        self.data = data
        self.data_missing = data_missing
        self.fully_defined_weighting = True # does the background dataset land in every leaf (making it valid for the tree_path_dependent method)
        self.tree_limit = None # used for limiting the number of trees we use by default (like from early stopping)
        self.num_stacked_models = 1 # If this is greater than 1 it means we have multiple stacked models with the same number of trees in each model (XGBoost multi-output style)
        self.cat_feature_indices = None # If this is set it tells us which features are treated categorically

        # we use names like keras
        objective_name_map = {
            "mse": "squared_error",
            "variance": "squared_error",
            "friedman_mse": "squared_error",
            "reg:linear": "squared_error",
            "reg:squarederror": "squared_error",
            "regression": "squared_error",
            "regression_l2": "squared_error",
            "mae": "absolute_error",
            "gini": "binary_crossentropy",
            "entropy": "binary_crossentropy",
            "reg:logistic": "binary_crossentropy",
            "binary:logistic": "binary_crossentropy",
            "binary_logloss": "binary_crossentropy",
            "binary": "binary_crossentropy"
        }

        tree_output_name_map = {
            "regression": "raw_value",
            "regression_l2": "squared_error",
            "reg:linear": "raw_value",
            "reg:squarederror": "raw_value",
            "reg:logistic": "log_odds",
            "binary:logistic": "log_odds",
            "binary_logloss": "log_odds",
            "binary": "log_odds"
        }

        if type(model) is dict and "trees" in model:
            # This allows a dictionary to be passed that represents the model.
            # this dictionary has several numerica paramters and also a list of trees
            # where each tree is a dictionary describing that tree
            if "internal_dtype" in model:
                self.internal_dtype = model["internal_dtype"]
            if "input_dtype" in model:
                self.input_dtype = model["input_dtype"]
            if "objective" in model:
                self.objective = model["objective"]
            if "tree_output" in model:
                self.tree_output = model["tree_output"]
            if "base_offset" in model:
                self.base_offset = model["base_offset"]
            self.trees = [SingleTree(t, data=data, data_missing=data_missing) for t in model["trees"]]
        elif type(model) is list and type(model[0]) == SingleTree: # old-style direct-load format
            self.trees = model
        elif safe_isinstance(model, ["sklearn.ensemble.RandomForestRegressor", "sklearn.ensemble.forest.RandomForestRegressor", "econml.grf._base_grf.BaseGRF"]):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(model, ["sklearn.ensemble.IsolationForest", "sklearn.ensemble._iforest.IsolationForest"]):
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [IsoTree(e.tree_, f, scaling=scaling, data=data, data_missing=data_missing) for e, f in zip(model.estimators_, model.estimators_features_)]
            self.tree_output = "raw_value"
        elif safe_isinstance(model, ["pyod.models.iforest.IForest"]):
            self.dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [IsoTree(e.tree_, f, scaling=scaling, data=data, data_missing=data_missing) for e, f in zip(model.detector_.estimators_, model.detector_.estimators_features_)]
            self.tree_output = "raw_value"
        elif safe_isinstance(model, "skopt.learning.forest.RandomForestRegressor"):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(model, ["sklearn.ensemble.ExtraTreesRegressor", "sklearn.ensemble.forest.ExtraTreesRegressor"]):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(model, "skopt.learning.forest.ExtraTreesRegressor"):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(model, ["sklearn.tree.DecisionTreeRegressor", "sklearn.tree.tree.DecisionTreeRegressor", "econml.grf._base_grftree.GRFTree"]):
            self.internal_dtype = model.tree_.value.dtype.type
            self.input_dtype = np.float32
            self.trees = [SingleTree(model.tree_, data=data, data_missing=data_missing)]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(model, ["sklearn.tree.DecisionTreeClassifier", "sklearn.tree.tree.DecisionTreeClassifier"]):
            self.internal_dtype = model.tree_.value.dtype.type
            self.input_dtype = np.float32
            self.trees = [SingleTree(model.tree_, normalize=True, data=data, data_missing=data_missing)]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif safe_isinstance(model, ["sklearn.ensemble.RandomForestClassifier", "sklearn.ensemble.forest.RandomForestClassifier"]):
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif safe_isinstance(model, ["sklearn.ensemble.ExtraTreesClassifier", "sklearn.ensemble.forest.ExtraTreesClassifier"]): # TODO: add unit test for this case
            assert hasattr(model, "estimators_"), "Model has no `estimators_`! Have you called `model.fit`?"
            self.internal_dtype = model.estimators_[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif safe_isinstance(model, ["sklearn.ensemble.GradientBoostingRegressor", "sklearn.ensemble.gradient_boosting.GradientBoostingRegressor"]):
            self.input_dtype = np.float32

            # currently we only support the mean and quantile estimators
            if safe_isinstance(model.init_, ["sklearn.ensemble.MeanEstimator", "sklearn.ensemble.gradient_boosting.MeanEstimator"]):
                self.base_offset = model.init_.mean
            elif safe_isinstance(model.init_, ["sklearn.ensemble.QuantileEstimator", "sklearn.ensemble.gradient_boosting.QuantileEstimator"]):
                self.base_offset = model.init_.quantile
            elif safe_isinstance(model.init_, "sklearn.dummy.DummyRegressor"):
                self.base_offset = model.init_.constant_[0]
            else:
                assert False, "Unsupported init model type: " + str(type(model.init_))

            self.trees = [SingleTree(e.tree_, scaling=model.learning_rate, data=data, data_missing=data_missing) for e in model.estimators_[:,0]]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(model, ["sklearn.ensemble.HistGradientBoostingRegressor"]):
            import sklearn
            if self.model_output == "predict":
                self.model_output = "raw"
            self.input_dtype = sklearn.ensemble._hist_gradient_boosting.common.X_DTYPE
            self.base_offset = model._baseline_prediction
            self.trees = []
            for p in model._predictors:
                nodes = p[0].nodes
                # each node has values: ('value', 'count', 'feature_idx', 'threshold', 'missing_go_to_left', 'left', 'right', 'gain', 'depth', 'is_leaf', 'bin_threshold')
                tree = {
                    "children_left": np.array([-1 if n[9] else n[5] for n in nodes]),
                    "children_right": np.array([-1 if n[9] else n[6] for n in nodes]),
                    "children_default": np.array([-1 if n[9] else (n[5] if n[4] else n[6]) for n in nodes]),
                    "features": np.array([-2 if n[9] else n[2] for n in nodes]),
                    "thresholds": np.array([n[3] for n in nodes], dtype=np.float64),
                    "values": np.array([[n[0]] for n in nodes], dtype=np.float64),
                    "node_sample_weight": np.array([n[1] for n in nodes], dtype=np.float64),
                }
                self.trees.append(SingleTree(tree, data=data, data_missing=data_missing))
            self.objective = objective_name_map.get(model.loss, None)
            self.tree_output = "raw_value"
        elif safe_isinstance(model, ["sklearn.ensemble.HistGradientBoostingClassifier"]):
            import sklearn
            self.base_offset = model._baseline_prediction
            if hasattr(self.base_offset, "__len__") and self.model_output != "raw":
                raise Exception("Multi-output HistGradientBoostingClassifier models are not yet supported unless model_output=\"raw\". See GitHub issue #1028")
            self.input_dtype = sklearn.ensemble._hist_gradient_boosting.common.X_DTYPE
            self.num_stacked_models = len(model._predictors[0])
            if self.model_output == "predict_proba":
                if self.num_stacked_models == 1:
                    self.model_output = "probability_doubled" # with predict_proba we need to double the outputs to match
                else:
                    self.model_output = "probability"
            self.trees = []
            for p in model._predictors:
                for i in range(self.num_stacked_models):
                    nodes = p[i].nodes
                    # each node has values: ('value', 'count', 'feature_idx', 'threshold', 'missing_go_to_left', 'left', 'right', 'gain', 'depth', 'is_leaf', 'bin_threshold')
                    tree = {
                        "children_left": np.array([-1 if n[9] else n[5] for n in nodes]),
                        "children_right": np.array([-1 if n[9] else n[6] for n in nodes]),
                        "children_default": np.array([-1 if n[9] else (n[5] if n[4] else n[6]) for n in nodes]),
                        "features": np.array([-2 if n[9] else n[2] for n in nodes]),
                        "thresholds": np.array([n[3] for n in nodes], dtype=np.float64),
                        "values": np.array([[n[0]] for n in nodes], dtype=np.float64),
                        "node_sample_weight": np.array([n[1] for n in nodes], dtype=np.float64),
                    }
                    self.trees.append(SingleTree(tree, data=data, data_missing=data_missing))
            self.objective = objective_name_map.get(model.loss, None)
            self.tree_output = "log_odds"
        elif safe_isinstance(model, ["sklearn.ensemble.GradientBoostingClassifier","sklearn.ensemble._gb.GradientBoostingClassifier", "sklearn.ensemble.gradient_boosting.GradientBoostingClassifier"]):
            self.input_dtype = np.float32

            # TODO: deal with estimators for each class
            if model.estimators_.shape[1] > 1:
                assert False, "GradientBoostingClassifier is only supported for binary classification right now!"

            # currently we only support the logs odds estimator
            if safe_isinstance(model.init_, ["sklearn.ensemble.LogOddsEstimator", "sklearn.ensemble.gradient_boosting.LogOddsEstimator"]):
                self.base_offset = model.init_.prior
                self.tree_output = "log_odds"
            elif safe_isinstance(model.init_, "sklearn.dummy.DummyClassifier"):
                self.base_offset = scipy.special.logit(model.init_.class_prior_[1]) # with two classes the trees only model the second class. # pylint: disable=no-member
                self.tree_output = "log_odds"
            else:
                assert False, "Unsupported init model type: " + str(type(model.init_))

            self.trees = [SingleTree(e.tree_, scaling=model.learning_rate, data=data, data_missing=data_missing) for e in model.estimators_[:,0]]
            self.objective = objective_name_map.get(model.criterion, None)
        elif "pyspark.ml" in str(type(model)):
            assert_import("pyspark")
            self.model_type = "pyspark"
            # model._java_obj.getImpurity() can be gini, entropy or variance.
            self.objective = objective_name_map.get(model._java_obj.getImpurity(), None)
            if "Classification" in str(type(model)):
                normalize = True
                self.tree_output = "probability"
            else:
                normalize = False
                self.tree_output = "raw_value"
            # Spark Random forest, create 1 weighted (avg) tree per sub-model
            if safe_isinstance(model, "pyspark.ml.classification.RandomForestClassificationModel") \
                    or safe_isinstance(model, "pyspark.ml.regression.RandomForestRegressionModel"):
                sum_weight = sum(model.treeWeights)  # output is average of trees
                self.trees = [SingleTree(tree, normalize=normalize, scaling=model.treeWeights[i]/sum_weight) for i, tree in enumerate(model.trees)]
            # Spark GBT, create 1 weighted (learning rate) tree per sub-model
            elif safe_isinstance(model, "pyspark.ml.classification.GBTClassificationModel") \
                    or safe_isinstance(model, "pyspark.ml.regression.GBTRegressionModel"):
                self.objective = "squared_error" # GBT subtree use the variance
                self.tree_output = "raw_value"
                self.trees = [SingleTree(tree, normalize=False, scaling=model.treeWeights[i]) for i, tree in enumerate(model.trees)]
            # Spark Basic model (single tree)
            elif safe_isinstance(model, "pyspark.ml.classification.DecisionTreeClassificationModel") \
                    or safe_isinstance(model, "pyspark.ml.regression.DecisionTreeRegressionModel"):
                self.trees = [SingleTree(model, normalize=normalize, scaling=1)]
            else:
                assert False, "Unsupported Spark model type: " + str(type(model))
        elif safe_isinstance(model, "xgboost.core.Booster"):
            import xgboost
            self.original_model = model
            self.model_type = "xgboost"
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            self.objective = objective_name_map.get(xgb_loader.name_obj, None)
            self.tree_output = tree_output_name_map.get(xgb_loader.name_obj, None)
            if xgb_loader.num_class > 0:
                self.num_stacked_models = xgb_loader.num_class
        elif safe_isinstance(model, "xgboost.sklearn.XGBClassifier"):
            import xgboost
            self.input_dtype = np.float32
            self.model_type = "xgboost"
            self.original_model = model.get_booster()
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            self.objective = objective_name_map.get(xgb_loader.name_obj, None)
            self.tree_output = tree_output_name_map.get(xgb_loader.name_obj, None)
            self.tree_limit = getattr(model, "best_ntree_limit", None)
            if xgb_loader.num_class > 0:
                self.num_stacked_models = xgb_loader.num_class
            if self.model_output == "predict_proba":
                if self.num_stacked_models == 1:
                    self.model_output = "probability_doubled" # with predict_proba we need to double the outputs to match
                else:
                    self.model_output = "probability"
        elif safe_isinstance(model, "xgboost.sklearn.XGBRegressor"):
            import xgboost
            self.original_model = model.get_booster()
            self.model_type = "xgboost"
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            self.objective = objective_name_map.get(xgb_loader.name_obj, None)
            self.tree_output = tree_output_name_map.get(xgb_loader.name_obj, None)
            self.tree_limit = getattr(model, "best_ntree_limit", None)
            if xgb_loader.num_class > 0:
                self.num_stacked_models = xgb_loader.num_class
        elif safe_isinstance(model, "xgboost.sklearn.XGBRanker"):
            import xgboost
            self.original_model = model.get_booster()
            self.model_type = "xgboost"
            xgb_loader = XGBTreeModelLoader(self.original_model)
            self.trees = xgb_loader.get_trees(data=data, data_missing=data_missing)
            self.base_offset = xgb_loader.base_score
            # Note: for ranker, leaving tree_output and objective as None as they
            # are not implemented in native code yet
            self.tree_limit = getattr(model, "best_ntree_limit", None)
            if xgb_loader.num_class > 0:
                self.num_stacked_models = xgb_loader.num_class
        elif safe_isinstance(model, "lightgbm.basic.Booster"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except:
                self.trees = None # we get here because the cext can't handle categorical splits yet

            self.objective = objective_name_map.get(model.params.get("objective", "regression"), None)
            self.tree_output = tree_output_name_map.get(model.params.get("objective", "regression"), None)

        elif safe_isinstance(model, "gpboost.basic.Booster"):
            assert_import("gpboost")
            self.model_type = "gpboost"
            self.original_model = model
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except:
                self.trees = None # we get here because the cext can't handle categorical splits yet

            self.objective = objective_name_map.get(model.params.get("objective", "regression"), None)
            self.tree_output = tree_output_name_map.get(model.params.get("objective", "regression"), None)

        elif safe_isinstance(model, "lightgbm.sklearn.LGBMRegressor"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model.booster_
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except:
                self.trees = None # we get here because the cext can't handle categorical splits yet
            self.objective = objective_name_map.get(model.objective, None)
            self.tree_output = tree_output_name_map.get(model.objective, None)
            if model.objective is None:
                self.objective = "squared_error"
                self.tree_output = "raw_value"
        elif safe_isinstance(model, "lightgbm.sklearn.LGBMRanker"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            self.original_model = model.booster_
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except:
                self.trees = None # we get here because the cext can't handle categorical splits yet
            # Note: for ranker, leaving tree_output and objective as None as they
            # are not implemented in native code yet
        elif safe_isinstance(model, "lightgbm.sklearn.LGBMClassifier"):
            assert_import("lightgbm")
            self.model_type = "lightgbm"
            if model.n_classes_ > 2:
                self.num_stacked_models = model.n_classes_
            self.original_model = model.booster_
            tree_info = self.original_model.dump_model()["tree_info"]
            try:
                self.trees = [SingleTree(e, data=data, data_missing=data_missing) for e in tree_info]
            except:
                self.trees = None # we get here because the cext can't handle categorical splits yet
            self.objective = objective_name_map.get(model.objective, None)
            self.tree_output = tree_output_name_map.get(model.objective, None)
            if model.objective is None:
                self.objective = "binary_crossentropy"
                self.tree_output = "log_odds"
        elif safe_isinstance(model, "catboost.core.CatBoostRegressor"):
            assert_import("catboost")
            self.model_type = "catboost"
            self.original_model = model
            self.cat_feature_indices = model.get_cat_feature_indices()
        elif safe_isinstance(model, "catboost.core.CatBoostClassifier"):
            assert_import("catboost")
            self.model_type = "catboost"
            self.original_model = model
            self.input_dtype = np.float32
            try:
                cb_loader = CatBoostTreeModelLoader(model)
                self.trees = cb_loader.get_trees(data=data, data_missing=data_missing)
            except:
                self.trees = None # we get here because the cext can't handle categorical splits yet
            self.tree_output = "log_odds"
            self.objective = "binary_crossentropy"
            self.cat_feature_indices = model.get_cat_feature_indices()
        elif safe_isinstance(model, "catboost.core.CatBoost"):
            assert_import("catboost")
            self.model_type = "catboost"
            self.original_model = model
            self.cat_feature_indices = model.get_cat_feature_indices()
        elif safe_isinstance(model, "imblearn.ensemble._forest.BalancedRandomForestClassifier"):
            self.input_dtype = np.float32
            scaling = 1.0 / len(model.estimators_) # output is average of trees
            self.trees = [SingleTree(e.tree_, normalize=True, scaling=scaling, data=data, data_missing=data_missing) for e in model.estimators_]
            self.objective = objective_name_map.get(model.criterion, None)
            self.tree_output = "probability"
        elif safe_isinstance(model, "ngboost.ngboost.NGBoost") or safe_isinstance(model, "ngboost.api.NGBRegressor") or safe_isinstance(model, "ngboost.api.NGBClassifier"):
            assert model.base_models, "The NGBoost model has empty `base_models`! Have you called `model.fit`?"
            if self.model_output == "raw":
                param_idx = 0 # default to the first parameter of the output distribution
                warnings.warn("Translating model_ouput=\"raw\" to model_output=0 for the 0-th parameter in the distribution. Use model_output=0 directly to avoid this warning.")
            elif type(self.model_output) is int:
                param_idx = self.model_output
                self.model_output = "raw" # note that after loading we have a new model_output type
            assert safe_isinstance(model.base_models[0][param_idx], ["sklearn.tree.DecisionTreeRegressor", "sklearn.tree.tree.DecisionTreeRegressor"]), "You must use default_tree_learner!"
            shap_trees = [trees[param_idx] for trees in model.base_models]
            self.internal_dtype = shap_trees[0].tree_.value.dtype.type
            self.input_dtype = np.float32
            scaling = - model.learning_rate * np.array(model.scalings) # output is weighted average of trees
            self.trees = [SingleTree(e.tree_, scaling=s, data=data, data_missing=data_missing) for e,s in zip(shap_trees,scaling)]
            self.objective = objective_name_map.get(shap_trees[0].criterion, None)
            self.tree_output = "raw_value"
            self.base_offset = model.init_params[param_idx]
        else:
            raise Exception("Model type not yet supported by TreeExplainer: " + str(type(model)))

        # build a dense numpy version of all the tree objects
        if self.trees is not None and self.trees:
            max_nodes = np.max([len(t.values) for t in self.trees])
            assert len(np.unique([t.values.shape[1] for t in self.trees])) == 1, "All trees in the ensemble must have the same output dimension!"
            num_trees = len(self.trees)
            if self.num_stacked_models > 1:
                assert len(self.trees) % self.num_stacked_models == 0, "Only stacked models with equal numbers of trees are supported!"
                assert self.trees[0].values.shape[1] == 1, "Only stacked models with single outputs per model are supported!"
                self.num_outputs = self.num_stacked_models
            else:
                self.num_outputs = self.trees[0].values.shape[1]

            # important to be -1 in unused sections!! This way we can tell which entries are valid.
            self.children_left = -np.ones((num_trees, max_nodes), dtype=np.int32)
            self.children_right = -np.ones((num_trees, max_nodes), dtype=np.int32)
            self.children_default = -np.ones((num_trees, max_nodes), dtype=np.int32)
            self.features = -np.ones((num_trees, max_nodes), dtype=np.int32)

            self.thresholds = np.zeros((num_trees, max_nodes), dtype=self.internal_dtype)
            self.values = np.zeros((num_trees, max_nodes, self.num_outputs), dtype=self.internal_dtype)
            self.node_sample_weight = np.zeros((num_trees, max_nodes), dtype=self.internal_dtype)

            for i in range(num_trees):
                self.children_left[i,:len(self.trees[i].children_left)] = self.trees[i].children_left
                self.children_right[i,:len(self.trees[i].children_right)] = self.trees[i].children_right
                self.children_default[i,:len(self.trees[i].children_default)] = self.trees[i].children_default
                self.features[i,:len(self.trees[i].features)] = self.trees[i].features
                self.thresholds[i,:len(self.trees[i].thresholds)] = self.trees[i].thresholds
                if self.num_stacked_models > 1:
                    # stack_pos = int(i // (num_trees / self.num_stacked_models))
                    stack_pos = i % self.num_stacked_models
                    self.values[i,:len(self.trees[i].values[:,0]),stack_pos] = self.trees[i].values[:,0]
                else:
                    self.values[i,:len(self.trees[i].values)] = self.trees[i].values
                self.node_sample_weight[i,:len(self.trees[i].node_sample_weight)] = self.trees[i].node_sample_weight

                # ensure that the passed background dataset lands in every leaf
                if np.min(self.trees[i].node_sample_weight) <= 0:
                    self.fully_defined_weighting = False

            self.num_nodes = np.array([len(t.values) for t in self.trees], dtype=np.int32)
            self.max_depth = np.max([t.max_depth for t in self.trees])

            # make sure the base offset is a 1D array
            if not hasattr(self.base_offset, "__len__") or len(self.base_offset) == 0:
                self.base_offset = (np.ones(self.num_outputs) * self.base_offset).astype(self.internal_dtype)
            self.base_offset = self.base_offset.flatten()
            assert len(self.base_offset) == self.num_outputs

    def get_transform(self):
        """ A consistent interface to make predictions from this model.
        """
        if self.model_output == "raw":
            transform = "identity"
        elif self.model_output == "probability" or self.model_output == "probability_doubled":
            if self.tree_output == "log_odds":
                transform = "logistic"
            elif self.tree_output == "probability":
                transform = "identity"
            else:
                raise Exception("model_output = \"probability\" is not yet supported when model.tree_output = \"" + self.tree_output + "\"!")
        elif self.model_output == "log_loss":

            if self.objective == "squared_error":
                transform = "squared_loss"
            elif self.objective == "binary_crossentropy":
                transform = "logistic_nlogloss"
            else:
                raise Exception("model_output = \"log_loss\" is not yet supported when model.objective = \"" + self.objective + "\"!")
        else:
            raise Exception("Unrecognized model_output parameter value: %s! If model.%s is a valid function open a github issue to ask that this method be supported. If you want 'predict_proba' just use 'probability' for now." % (str(self.model_output), str(self.model_output)))

        return transform

    def predict(self, X, y=None, output=None, tree_limit=None):
        """ A consistent interface to make predictions from this model.

        Parameters
        ----------
        tree_limit : None (default) or int
            Limit the number of trees used by the model. By default None means no use the limit of the
            original model, and -1 means no limit.
        """

        if output is None:
            output = self.model_output

        if self.model_type == "pyspark":
            #import pyspark
            # TODO: support predict for pyspark
            raise NotImplementedError("Predict with pyspark isn't implemented. Don't run 'interventional' as feature_perturbation.")

        # see if we have a default tree_limit in place.
        if tree_limit is None:
            tree_limit = -1 if self.tree_limit is None else self.tree_limit

        # convert dataframes
        if safe_isinstance(X, "pandas.core.series.Series"):
            X = X.values
        elif safe_isinstance(X, "pandas.core.frame.DataFrame"):
            X = X.values
        flat_output = False
        if len(X.shape) == 1:
            flat_output = True
            X = X.reshape(1, X.shape[0])
        if X.dtype.type != self.input_dtype:
            X = X.astype(self.input_dtype)
        X_missing = np.isnan(X, dtype=np.bool)
        assert isinstance(X, np.ndarray), "Unknown instance type: " + str(type(X))
        assert len(X.shape) == 2, "Passed input data matrix X must have 1 or 2 dimensions!"

        if tree_limit < 0 or tree_limit > self.values.shape[0]:
            tree_limit = self.values.shape[0]

        if output == "logloss":
            assert y is not None, "Both samples and labels must be provided when explaining the loss (i.e. `explainer.shap_values(X, y)`)!"
            assert X.shape[0] == len(y), "The number of labels (%d) does not match the number of samples to explain (%d)!" % (len(y), X.shape[0])
        transform = self.get_transform()
        assert_import("cext")
        output = np.zeros((X.shape[0], self.num_outputs))
        _cext.dense_tree_predict(
            self.children_left, self.children_right, self.children_default,
            self.features, self.thresholds, self.values,
            self.max_depth, tree_limit, self.base_offset, output_transform_codes[transform],
            X, X_missing, y, output
        )

        # drop dimensions we don't need
        if flat_output:
            if self.num_outputs == 1:
                return output.flatten()[0]
            else:
                return output.reshape(-1, self.num_outputs)
        else:
            if self.num_outputs == 1:
                return output.flatten()
            else:
                return output


class XGBTreeModelLoader(object):
    """ This loads an XGBoost model directly from a raw memory dump.

    We can't use the JSON dump because due to numerical precision issues those
    tree can actually be wrong when feature values land almost on a threshold.
    """
    def __init__(self, xgb_model):
        # new in XGBoost 1.1, 'binf' is appended to the buffer
        self.buf = xgb_model.save_raw().lstrip(b'binf')
        self.pos = 0

        # load the model parameters
        self.base_score = self.read('f')
        self.num_feature = self.read('I')
        self.num_class = self.read('i')
        self.contain_extra_attrs = self.read('i')
        self.contain_eval_metrics = self.read('i')
        self.read_arr('i', 29) # reserved
        self.name_obj_len = self.read('Q')
        self.name_obj = self.read_str(self.name_obj_len)
        self.name_gbm_len = self.read('Q')
        self.name_gbm = self.read_str(self.name_gbm_len)

        # new in XGBoost 1.0 is that the base_score is saved untransformed (https://github.com/dmlc/xgboost/pull/5101)
        # so we have to transform it depending on the objective
        import xgboost
        if LooseVersion(xgboost.__version__).version[0] >= 1:
            if self.name_obj in ["binary:logistic", "reg:logistic"]:
                self.base_score = scipy.special.logit(self.base_score) # pylint: disable=no-member

        assert self.name_gbm == "gbtree", "Only the 'gbtree' model type is supported, not '%s'!" % self.name_gbm

        # load the gbtree specific parameters
        self.num_trees = self.read('i')
        self.num_roots = self.read('i')
        self.num_feature = self.read('i')
        self.pad_32bit = self.read('i')
        self.num_pbuffer_deprecated = self.read('Q')
        self.num_output_group = self.read('i')
        self.size_leaf_vector = self.read('i')
        self.read_arr('i', 32) # reserved

        # load each tree
        self.num_roots = np.zeros(self.num_trees, dtype=np.int32)
        self.num_nodes = np.zeros(self.num_trees, dtype=np.int32)
        self.num_deleted = np.zeros(self.num_trees, dtype=np.int32)
        self.max_depth = np.zeros(self.num_trees, dtype=np.int32)
        self.num_feature = np.zeros(self.num_trees, dtype=np.int32)
        self.size_leaf_vector = np.zeros(self.num_trees, dtype=np.int32)
        self.node_parents = []
        self.node_cleft = []
        self.node_cright = []
        self.node_sindex = []
        self.node_info = []
        self.loss_chg = []
        self.sum_hess = []
        self.base_weight = []
        self.leaf_child_cnt = []
        for i in range(self.num_trees):

            # load the per-tree params
            self.num_roots[i] = self.read('i')
            self.num_nodes[i] = self.read('i')
            self.num_deleted[i] = self.read('i')
            self.max_depth[i] = self.read('i')
            self.num_feature[i] = self.read('i')
            self.size_leaf_vector[i] = self.read('i')

            # load the nodes
            self.read_arr('i', 31) # reserved
            self.node_parents.append(np.zeros(self.num_nodes[i], dtype=np.int32))
            self.node_cleft.append(np.zeros(self.num_nodes[i], dtype=np.int32))
            self.node_cright.append(np.zeros(self.num_nodes[i], dtype=np.int32))
            self.node_sindex.append(np.zeros(self.num_nodes[i], dtype=np.uint32))
            self.node_info.append(np.zeros(self.num_nodes[i], dtype=np.float32))
            for j in range(self.num_nodes[i]):
                self.node_parents[-1][j] = self.read('i')
                self.node_cleft[-1][j] = self.read('i')
                self.node_cright[-1][j] = self.read('i')
                self.node_sindex[-1][j] = self.read('I')
                self.node_info[-1][j] = self.read('f')

            # load the stat nodes
            self.loss_chg.append(np.zeros(self.num_nodes[i], dtype=np.float32))
            self.sum_hess.append(np.zeros(self.num_nodes[i], dtype=np.float32))
            self.base_weight.append(np.zeros(self.num_nodes[i], dtype=np.float32))
            self.leaf_child_cnt.append(np.zeros(self.num_nodes[i], dtype=np.int))
            for j in range(self.num_nodes[i]):
                self.loss_chg[-1][j] = self.read('f')
                self.sum_hess[-1][j] = self.read('f')
                self.base_weight[-1][j] = self.read('f')
                self.leaf_child_cnt[-1][j] = self.read('i')

    def get_trees(self, data=None, data_missing=None):
        shape = (self.num_trees, self.num_nodes.max())
        self.children_default = np.zeros(shape, dtype=np.int)
        self.features = np.zeros(shape, dtype=np.int)
        self.thresholds = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros((shape[0], shape[1], 1), dtype=np.float32)
        trees = []
        for i in range(self.num_trees):
            for j in range(self.num_nodes[i]):
                if np.right_shift(self.node_sindex[i][j], np.uint32(31)) != 0:
                    self.children_default[i,j] = self.node_cleft[i][j]
                else:
                    self.children_default[i,j] = self.node_cright[i][j]
                self.features[i,j] = self.node_sindex[i][j] & ((np.uint32(1) << np.uint32(31)) - np.uint32(1))
                if self.node_cleft[i][j] >= 0:
                    # Xgboost uses < for thresholds where shap uses <=
                    # Move the threshold down by the smallest possible increment
                    self.thresholds[i, j] = np.nextafter(self.node_info[i][j], - np.float32(np.inf))
                else:
                    self.values[i,j] = self.node_info[i][j]

            l = len(self.node_cleft[i])
            trees.append(SingleTree({
                "children_left": self.node_cleft[i],
                "children_right": self.node_cright[i],
                "children_default": self.children_default[i,:l],
                "feature": self.features[i,:l],
                "threshold": self.thresholds[i,:l],
                "value": self.values[i,:l],
                "node_sample_weight": self.sum_hess[i]
            }, data=data, data_missing=data_missing))
        return trees


    def read(self, dtype):
        size = struct.calcsize(dtype)
        val = struct.unpack(dtype, self.buf[self.pos:self.pos+size])[0]
        self.pos += size
        return val

    def read_arr(self, dtype, n_items):
        format = "%d%s" % (n_items, dtype)
        size = struct.calcsize(format)
        val = struct.unpack(format, self.buf[self.pos:self.pos+size])[0]
        self.pos += size
        return val

    def read_str(self, size):
        val = self.buf[self.pos:self.pos+size].decode('utf-8')
        self.pos += size
        return val

    def print_info(self):

        print("--- global parmeters ---")
        print("base_score =", self.base_score)
        print("num_feature =", self.num_feature)
        print("num_class =", self.num_class)
        print("contain_extra_attrs =", self.contain_extra_attrs)
        print("contain_eval_metrics =", self.contain_eval_metrics)
        print("name_obj_len =", self.name_obj_len)
        print("name_obj =", self.name_obj)
        print("name_gbm_len =", self.name_gbm_len)
        print("name_gbm =", self.name_gbm)
        print()
        print("--- gbtree specific parameters ---")
        print("num_trees =", self.num_trees)
        print("num_roots =", self.num_roots)
        print("num_feature =", self.num_feature)
        print("pad_32bit =", self.pad_32bit)
        print("num_pbuffer_deprecated =", self.num_pbuffer_deprecated)
        print("num_output_group =", self.num_output_group)
        print("size_leaf_vector =", self.size_leaf_vector)


class CatBoostTreeModelLoader:
    def __init__(self, cb_model):
        # cb_model.save_model("cb_model.json", format="json")
        # self.loaded_cb_model = json.load(open("cb_model.json", "r"))
        import tempfile
        tmp_file = tempfile.NamedTemporaryFile()
        cb_model.save_model(tmp_file.name, format="json")
        self.loaded_cb_model = json.load(open(tmp_file.name, "r"))
        tmp_file.close()

        # load the CatBoost oblivious trees specific parameters
        self.num_trees = len(self.loaded_cb_model['oblivious_trees'])
        self.max_depth = self.loaded_cb_model['model_info']['params']['tree_learner_options']['depth']

    def get_trees(self, data=None, data_missing=None):
        # load each tree
        trees = []
        for tree_index in range(self.num_trees):

            # load the per-tree params
            #depth = len(self.loaded_cb_model['oblivious_trees'][tree_index]['splits'])

            # load the nodes

            # Re-compute the number of samples that pass through each node if we are given data
            leaf_weights = self.loaded_cb_model['oblivious_trees'][tree_index]['leaf_weights']
            leaf_weights_unraveled = [0] * (len(leaf_weights) - 1) + leaf_weights
            leaf_weights_unraveled[0] = sum(leaf_weights)
            for index in range(len(leaf_weights) - 2, 0, -1):
                leaf_weights_unraveled[index] = leaf_weights_unraveled[2 * index + 1] + leaf_weights_unraveled[2 * index + 2]

            leaf_values = self.loaded_cb_model['oblivious_trees'][tree_index]['leaf_values']
            leaf_values_unraveled = [0] * (len(leaf_values) - 1) + leaf_values

            children_left = [i * 2 + 1 for i in range(len(leaf_values) - 1)]
            children_left += [-1] * len(leaf_values)

            children_right = [i * 2 for i in range(1, len(leaf_values))]
            children_right += [-1] * len(leaf_values)

            children_default = [i * 2 + 1 for i in range(len(leaf_values) - 1)]
            children_default += [-1] * len(leaf_values)

            # load the split features and borders
            # split features and borders go from leafs to the root
            split_features_index = []
            borders = []

            # split features and borders go from leafs to the root
            for elem in self.loaded_cb_model['oblivious_trees'][tree_index]['splits']:
                split_type = elem.get('split_type')
                if split_type == 'FloatFeature':
                    split_feature_index = elem.get('float_feature_index')
                    borders.append(elem['border'])
                elif split_type == 'OneHotFeature':
                    split_feature_index = elem.get('cat_feature_index')
                    borders.append(elem['value'])
                else:
                    split_feature_index = elem.get('ctr_target_border_idx')
                    borders.append(elem['border'])
                split_features_index.append(split_feature_index)

            split_features_index_unraveled = []
            for counter, feature_index in enumerate(split_features_index[::-1]):
                split_features_index_unraveled += [feature_index] * (2 ** counter)
            split_features_index_unraveled += [0] * len(leaf_values)

            borders_unraveled = []
            for counter, border in enumerate(borders[::-1]):
                borders_unraveled += [border] * (2 ** counter)
            borders_unraveled += [0] * len(leaf_values)

            trees.append(SingleTree({"children_left": np.array(children_left),
                                     "children_right": np.array(children_right),
                                     "children_default": np.array(children_default),
                                     "feature": np.array(split_features_index_unraveled),
                                     "threshold": np.array(borders_unraveled),
                                     "value": np.array(leaf_values_unraveled).reshape((-1,1)),
                                     "node_sample_weight": np.array(leaf_weights_unraveled),
                                     }, data=data, data_missing=data_missing))

        return trees
