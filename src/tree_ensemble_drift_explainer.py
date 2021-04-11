import numpy as np
from distutils.version import LooseVersion
from ._explainer import Explainer
from .tree_ensemble import TreeEnsemble
from ..utils import assert_import, record_import_error, safe_isinstance
from ..utils._legacy import DenseData
from .._explanation import Explanation
from .. import maskers
import warnings
import pandas as pd



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

class TreeEnsembleDriftExplainer(Explainer): # TODO : must inherit from DriftExplainer at the end
    """ Uses Tree SHAP algorithms to explain the output of ensemble tree models.

    Tree SHAP is a fast and exact method to estimate SHAP values for tree models and ensembles of trees,
    under several different possible assumptions about feature dependence. It depends on fast C++
    implementations either inside an externel model package or in the local compiled C extention.
    """

    def __init__(self, model, data = None, model_output="raw", feature_perturbation="interventional", feature_names=None, **deprecated_options):
        """ Build a new Tree explainer for the passed model.

        Parameters
        ----------
        model : model object
            The tree based machine learning model that we want to explain. XGBoost, LightGBM, CatBoost, Pyspark
            and most tree-based scikit-learn models are supported.

        data : numpy.array or pandas.DataFrame
            The background dataset to use for integrating out features. This argument is optional when
            feature_perturbation="tree_path_dependent", since in that case we can use the number of training
            samples that went down each tree path as our background dataset (this is recorded in the model object).

        feature_perturbation : "interventional" (default) or "tree_path_dependent" (default when data=None)
            Since SHAP values rely on conditional expectations we need to decide how to handle correlated
            (or otherwise dependent) input features. The "interventional" approach breaks the dependencies between
            features according to the rules dictated by causal inference (Janzing et al. 2019). Note that the
            "interventional" option requires a background dataset and its runtime scales linearly with the size
            of the background dataset you use. Anywhere from 100 to 1000 random background samples are good
            sizes to use. The "tree_path_dependent" approach is to just follow the trees and use the number
            of training examples that went down each leaf to represent the background distribution. This approach
            does not require a background dataset and so is used by default when no background dataset is provided.

        model_output : "raw", "probability", "log_loss", or model method name
            What output of the model should be explained. If "raw" then we explain the raw output of the
            trees, which varies by model. For regression models "raw" is the standard output, for binary
            classification in XGBoost this is the log odds ratio. If model_output is the name of a supported
            prediction method on the model object then we explain the output of that model method name.
            For example model_output="predict_proba" explains the result of calling model.predict_proba.
            If "probability" then we explain the output of the model transformed into probability space
            (note that this means the SHAP values now sum to the probability output of the model). If "logloss"
            then we explain the log base e of the model loss function, so that the SHAP values sum up to the
            log loss of the model for each sample. This is helpful for breaking down model performance by feature.
            Currently the probability and logloss options are only supported when feature_dependence="independent".

        Examples
        --------
        See `Tree explainer examples <https://shap.readthedocs.io/en/latest/api_examples/explainers/Tree.html>`_
        """
        if feature_names is not None:
            self.data_feature_names=feature_names
        elif safe_isinstance(data, "pandas.core.frame.DataFrame"):
            self.data_feature_names = list(data.columns)

        masker = data
        super(Tree, self).__init__(model, masker, feature_names=feature_names)

        if type(self.masker) is maskers.Independent:
            data = self.masker.data
        elif masker is not None:
            raise Exception("Unsupported masker type: %s!" % str(type(self.masker)))

        if getattr(self.masker, "clustering", None) is not None:
            raise Exception("TreeExplainer does not support clustered data inputs! Please use shap.Explainer or pass an unclustered masker!")

        # check for deprecated options
        if model_output == "margin":
            warnings.warn("model_output = \"margin\" has been renamed to model_output = \"raw\"")
            model_output = "raw"
        if model_output == "logloss":
            warnings.warn("model_output = \"logloss\" has been renamed to model_output = \"log_loss\"")
            model_output = "log_loss"
        if "feature_dependence" in deprecated_options:
            dep_val = deprecated_options["feature_dependence"]
            if dep_val == "independent" and feature_perturbation == "interventional":
                warnings.warn("feature_dependence = \"independent\" has been renamed to feature_perturbation" \
                              " = \"interventional\"! See GitHub issue #882.")
            elif feature_perturbation != "interventional":
                warnings.warn("feature_dependence = \"independent\" has been renamed to feature_perturbation" \
                              " = \"interventional\", you can't supply both options! See GitHub issue #882.")
            if dep_val == "tree_path_dependent" and feature_perturbation == "interventional":
                raise Exception("The feature_dependence option has been renamed to feature_perturbation! " \
                                "Please update the option name before calling TreeExplainer. See GitHub issue #882.")
        if feature_perturbation == "independent":
            raise Exception("feature_perturbation = \"independent\" is not a valid option value, please use " \
                            "feature_perturbation = \"interventional\" instead. See GitHub issue #882.")


        if safe_isinstance(data, "pandas.core.frame.DataFrame"):
            self.data = data.values
        elif isinstance(data, DenseData):
            self.data = data.data
        else:
            self.data = data
        if self.data is None:
            feature_perturbation = "tree_path_dependent"
            #warnings.warn("Setting feature_perturbation = \"tree_path_dependent\" because no background data was given.")
        elif feature_perturbation == "interventional" and self.data.shape[0] > 1000:
            warnings.warn("Passing "+str(self.data.shape[0]) + " background samples may lead to slow runtimes. Consider "
                                                               "using shap.sample(data, 100) to create a smaller background data set.")
        self.data_missing = None if self.data is None else pd.isna(self.data)
        self.feature_perturbation = feature_perturbation
        self.expected_value = None
        self.model = TreeEnsemble(model, self.data, self.data_missing, model_output)
        self.model_output = model_output
        #self.model_output = self.model.model_output # this allows the TreeEnsemble to translate model outputs types by how it loads the model

        if feature_perturbation not in feature_perturbation_codes:
            raise ValueError("Invalid feature_perturbation option!")

        # check for unsupported combinations of feature_perturbation and model_outputs
        if feature_perturbation == "tree_path_dependent":
            if self.model.model_output != "raw":
                raise ValueError("Only model_output=\"raw\" is supported for feature_perturbation=\"tree_path_dependent\"")
        elif data is None:
            raise ValueError("A background dataset must be provided unless you are using feature_perturbation=\"tree_path_dependent\"!")

        if self.model.model_output != "raw":
            if self.model.objective is None and self.model.tree_output is None:
                raise Exception("Model does not have a known objective or output type! When model_output is " \
                                "not \"raw\" then we need to know the model's objective or link function.")

        # A bug in XGBoost fixed in v0.81 makes XGBClassifier fail to give margin outputs
        if safe_isinstance(model, "xgboost.sklearn.XGBClassifier") and self.model.model_output != "raw":
            import xgboost
            if LooseVersion(xgboost.__version__) < LooseVersion('0.81'):
                raise RuntimeError("A bug in XGBoost fixed in v0.81 makes XGBClassifier fail to give margin outputs! Please upgrade to XGBoost >= v0.81!")

        # compute the expected value if we have a parsed tree for the cext
        if self.model.model_output == "log_loss":
            self.expected_value = self.__dynamic_expected_value
        elif data is not None:
            try:
                self.expected_value = self.model.predict(self.data).mean(0)
            except ValueError:
                raise Exception("Currently TreeExplainer can only handle models with categorical splits when " \
                                "feature_perturbation=\"tree_path_dependent\" and no background data is passed. Please try again using " \
                                "shap.TreeExplainer(model, feature_perturbation=\"tree_path_dependent\").")
            if hasattr(self.expected_value, '__len__') and len(self.expected_value) == 1:
                self.expected_value = self.expected_value[0]
        elif hasattr(self.model, "node_sample_weight"):
            self.expected_value = self.model.values[:,0].sum(0)
            if self.expected_value.size == 1:
                self.expected_value = self.expected_value[0]
            self.expected_value += self.model.base_offset
            if self.model.model_output != "raw":
                self.expected_value = None # we don't handle transforms in this case right now...

        # if our output format requires binary classification to be represented as two outputs then we do that here
        if self.model.model_output == "probability_doubled" and self.expected_value is not None:
            self.expected_value = [1-self.expected_value, self.expected_value]

    def __dynamic_expected_value(self, y):
        """ This computes the expected value conditioned on the given label value.
        """

        return self.model.predict(self.data, np.ones(self.data.shape[0]) * y).mean(0)

    def __call__(self, X, y=None, interactions=False, check_additivity=True):

        if safe_isinstance(X, "pandas.core.frame.DataFrame"):
            feature_names = list(X.columns)
            X = X.values
        else:
            feature_names = getattr(self, "data_feature_names", None)

        if not interactions:
            v = self.shap_values(X, y=y, from_call=True, check_additivity=check_additivity)
            output_shape = tuple()
            if type(v) is list:
                output_shape = (len(v),)
                v = np.stack(v, axis=-1) # put outputs at the end

            # the explanation object expects an expected value for each row
            if hasattr(self.expected_value, "__len__"):
                ev_tiled = np.tile(self.expected_value, (v.shape[0],1))
            else:
                ev_tiled = np.tile(self.expected_value, v.shape[0])

            e = Explanation(v, base_values=ev_tiled, data=X, feature_names=feature_names)
        else:
            v = self.shap_interaction_values(X)
            e = Explanation(v, base_values=self.expected_value, data=X, feature_names=feature_names, interaction_order=2)
        return e

    def _validate_inputs(self, X, y, tree_limit, check_additivity):
        # see if we have a default tree_limit in place.
        if tree_limit is None:
            tree_limit = -1 if self.model.tree_limit is None else self.model.tree_limit

        if tree_limit < 0 or tree_limit > self.model.values.shape[0]:
            tree_limit = self.model.values.shape[0]
        # convert dataframes
        if safe_isinstance(X, "pandas.core.series.Series"):
            X = X.values
        elif safe_isinstance(X, "pandas.core.frame.DataFrame"):
            X = X.values
        flat_output = False
        if len(X.shape) == 1:
            flat_output = True
            X = X.reshape(1, X.shape[0])
        if X.dtype != self.model.input_dtype:
            X = X.astype(self.model.input_dtype)
        X_missing = np.isnan(X, dtype=np.bool)
        assert isinstance(X, np.ndarray), "Unknown instance type: " + str(type(X))
        assert len(X.shape) == 2, "Passed input data matrix X must have 1 or 2 dimensions!"

        if self.model.model_output == "log_loss":
            assert y is not None, "Both samples and labels must be provided when model_output = " \
                                  "\"log_loss\" (i.e. `explainer.shap_values(X, y)`)!"
            assert X.shape[0] == len(
                y), "The number of labels (%d) does not match the number of samples to explain (" \
                    "%d)!" % (
                        len(y), X.shape[0])

        if self.feature_perturbation == "tree_path_dependent":
            assert self.model.fully_defined_weighting, "The background dataset you provided does " \
                                                       "not cover all the leaves in the model, " \
                                                       "so TreeExplainer cannot run with the " \
                                                       "feature_perturbation=\"tree_path_dependent\" option! " \
                                                       "Try providing a larger background " \
                                                       "dataset, or using " \
                                                       "feature_perturbation=\"interventional\"."

        if check_additivity and self.model.model_type == "pyspark":
            warnings.warn(
                "check_additivity requires us to run predictions which is not supported with "
                "spark, "
                "ignoring."
                " Set check_additivity=False to remove this warning")
            check_additivity = False

        return X, y, X_missing, flat_output, tree_limit, check_additivity


    def shap_values(self, X, y=None, tree_limit=None, approximate=False, check_additivity=True, from_call=False):
        """ Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array, pandas.DataFrame or catboost.Pool (for catboost)
            A matrix of samples (# samples x # features) on which to explain the model's output.

        y : numpy.array
            An array of label values for each sample. Used when explaining loss functions.

        tree_limit : None (default) or int
            Limit the number of trees used by the model. By default None means no use the limit of the
            original model, and -1 means no limit.

        approximate : bool
            Run fast, but only roughly approximate the Tree SHAP values. This runs a method
            previously proposed by Saabas which only considers a single feature ordering. Take care
            since this does not have the consistency guarantees of Shapley values and places too
            much weight on lower splits in the tree.

        check_additivity : bool
            Run a validation check that the sum of the SHAP values equals the output of the model. This
            check takes only a small amount of time, and will catch potential unforeseen errors.
            Note that this check only runs right now when explaining the margin of the model.

        Returns
        -------
        array or list
            For models with a single output this returns a matrix of SHAP values
            (# samples x # features). Each row sums to the difference between the model output for that
            sample and the expected value of the model output (which is stored in the expected_value
            attribute of the explainer when it is constant). For models with vector outputs this returns
            a list of such matrices, one for each output.
        """
        # see if we have a default tree_limit in place.
        if tree_limit is None:
            tree_limit = -1 if self.model.tree_limit is None else self.model.tree_limit

        # shortcut using the C++ version of Tree SHAP in XGBoost, LightGBM, and CatBoost
        if self.feature_perturbation == "tree_path_dependent" and self.model.model_type != "internal" and self.data is None:
            model_output_vals = None
            phi = None
            if self.model.model_type == "xgboost":
                import xgboost
                if not isinstance(X, xgboost.core.DMatrix):
                    X = xgboost.DMatrix(X)
                if tree_limit == -1:
                    tree_limit = 0
                try:
                    phi = self.model.original_model.predict(
                        X, ntree_limit=tree_limit, pred_contribs=True,
                        approx_contribs=approximate, validate_features=False
                    )
                except ValueError as e:
                    raise ValueError("This reshape error is often caused by passing a bad data matrix to SHAP. " \
                                     "See https://github.com/slundberg/shap/issues/580") from e

                if check_additivity and self.model.model_output == "raw":
                    model_output_vals = self.model.original_model.predict(
                        X, ntree_limit=tree_limit, output_margin=True,
                        validate_features=False
                    )

            elif self.model.model_type == "lightgbm":
                assert not approximate, "approximate=True is not supported for LightGBM models!"
                phi = self.model.original_model.predict(X, num_iteration=tree_limit, pred_contrib=True)
                # Note: the data must be joined on the last axis
                if self.model.original_model.params['objective'] == 'binary':
                    if not from_call:
                        warnings.warn('LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray')
                    phi = np.concatenate((0-phi, phi), axis=-1)
                if phi.shape[1] != X.shape[1] + 1:
                    try:
                        phi = phi.reshape(X.shape[0], phi.shape[1]//(X.shape[1]+1), X.shape[1]+1)
                    except ValueError as e:
                        raise Exception("This reshape error is often caused by passing a bad data matrix to SHAP. " \
                                        "See https://github.com/slundberg/shap/issues/580") from e

            elif self.model.model_type == "catboost": # thanks to the CatBoost team for implementing this...
                assert not approximate, "approximate=True is not supported for CatBoost models!"
                assert tree_limit == -1, "tree_limit is not yet supported for CatBoost models!"
                import catboost
                if type(X) != catboost.Pool:
                    X = catboost.Pool(X, cat_features=self.model.cat_feature_indices)
                phi = self.model.original_model.get_feature_importance(data=X, fstr_type='ShapValues')

            # note we pull off the last column and keep it as our expected_value
            if phi is not None:
                if len(phi.shape) == 3:
                    self.expected_value = [phi[0, i, -1] for i in range(phi.shape[1])]
                    out = [phi[:, i, :-1] for i in range(phi.shape[1])]
                else:
                    self.expected_value = phi[0, -1]
                    out = phi[:, :-1]

                if check_additivity and model_output_vals is not None:
                    self.assert_additivity(out, model_output_vals)

                return out

        X, y, X_missing, flat_output, tree_limit, check_additivity = self._validate_inputs(X, y,
                                                                                           tree_limit,
                                                                                           check_additivity)
        transform = self.model.get_transform()

        # run the core algorithm using the C extension
        assert_import("cext")
        phi = np.zeros((X.shape[0], X.shape[1]+1, self.model.num_outputs))
        if not approximate:
            _cext.dense_tree_shap(
                self.model.children_left, self.model.children_right, self.model.children_default,
                self.model.features, self.model.thresholds, self.model.values, self.model.node_sample_weight,
                self.model.max_depth, X, X_missing, y, self.data, self.data_missing, tree_limit,
                self.model.base_offset, phi, feature_perturbation_codes[self.feature_perturbation],
                output_transform_codes[transform], False
            )
        else:
            _cext.dense_tree_saabas(
                self.model.children_left, self.model.children_right, self.model.children_default,
                self.model.features, self.model.thresholds, self.model.values,
                self.model.max_depth, tree_limit, self.model.base_offset, output_transform_codes[transform],
                X, X_missing, y, phi
            )

        out = self._get_shap_output(phi, flat_output)
        if check_additivity and self.model.model_output == "raw":
            self.assert_additivity(out, self.model.predict(X))

        return out

    # we pull off the last column and keep it as our expected_value
    def _get_shap_output(self, phi, flat_output):
        if self.model.num_outputs == 1:
            if self.expected_value is None and self.model.model_output != "log_loss":
                self.expected_value = phi[0, -1, 0]
            if flat_output:
                out = phi[0, :-1, 0]
            else:
                out = phi[:, :-1, 0]
        else:
            if self.expected_value is None and self.model.model_output != "log_loss":
                self.expected_value = [phi[0, -1, i] for i in range(phi.shape[2])]
            if flat_output:
                out = [phi[0, :-1, i] for i in range(self.model.num_outputs)]
            else:
                out = [phi[:, :-1, i] for i in range(self.model.num_outputs)]


        # if our output format requires binary classificaiton to be represented as two outputs then we do that here
        if self.model.model_output == "probability_doubled":
            out = [-out, out]
        return out


    def shap_interaction_values(self, X, y=None, tree_limit=None):
        """ Estimate the SHAP interaction values for a set of samples.

        Parameters
        ----------
        X : numpy.array, pandas.DataFrame or catboost.Pool (for catboost)
            A matrix of samples (# samples x # features) on which to explain the model's output.

        y : numpy.array
            An array of label values for each sample. Used when explaining loss functions (not yet supported).

        tree_limit : None (default) or int
            Limit the number of trees used by the model. By default None means no use the limit of the
            original model, and -1 means no limit.

        Returns
        -------
        array or list
            For models with a single output this returns a tensor of SHAP values
            (# samples x # features x # features). The matrix (# features x # features) for each sample sums
            to the difference between the model output for that sample and the expected value of the model output
            (which is stored in the expected_value attribute of the explainer). Each row of this matrix sums to the
            SHAP value for that feature for that sample. The diagonal entries of the matrix represent the
            "main effect" of that feature on the prediction and the symmetric off-diagonal entries represent the
            interaction effects between all pairs of features for that sample. For models with vector outputs
            this returns a list of tensors, one for each output.
        """

        assert self.model.model_output == "raw", "Only model_output = \"raw\" is supported for SHAP interaction values right now!"
        #assert self.feature_perturbation == "tree_path_dependent", "Only feature_perturbation = \"tree_path_dependent\" is supported for SHAP interaction values right now!"
        transform = "identity"

        # see if we have a default tree_limit in place.
        if tree_limit is None:
            tree_limit = -1 if self.model.tree_limit is None else self.model.tree_limit

        # shortcut using the C++ version of Tree SHAP in XGBoost
        if self.model.model_type == "xgboost" and self.feature_perturbation == "tree_path_dependent":
            import xgboost
            if not isinstance(X, xgboost.core.DMatrix):
                X = xgboost.DMatrix(X)
            if tree_limit == -1:
                tree_limit = 0
            phi = self.model.original_model.predict(X, ntree_limit=tree_limit, pred_interactions=True, validate_features=False)

            # note we pull off the last column and keep it as our expected_value
            if len(phi.shape) == 4:
                self.expected_value = [phi[0, i, -1, -1] for i in range(phi.shape[1])]
                return [phi[:, i, :-1, :-1] for i in range(phi.shape[1])]
            else:
                self.expected_value = phi[0, -1, -1]
                return phi[:, :-1, :-1]

        X, y, X_missing, flat_output, tree_limit, _ = self._validate_inputs(X, y, tree_limit, False)
        # run the core algorithm using the C extension
        assert_import("cext")
        phi = np.zeros((X.shape[0], X.shape[1]+1, X.shape[1]+1, self.model.num_outputs))
        _cext.dense_tree_shap(
            self.model.children_left, self.model.children_right, self.model.children_default,
            self.model.features, self.model.thresholds, self.model.values, self.model.node_sample_weight,
            self.model.max_depth, X, X_missing, y, self.data, self.data_missing, tree_limit,
            self.model.base_offset, phi, feature_perturbation_codes[self.feature_perturbation],
            output_transform_codes[transform], True
        )

        return self._get_shap_interactions_output(phi,flat_output)

    # we pull off the last column and keep it as our expected_value
    def _get_shap_interactions_output(self, phi, flat_output):
        if self.model.num_outputs == 1:
            self.expected_value = phi[0, -1, -1, 0]
            if flat_output:
                out = phi[0, :-1, :-1, 0]
            else:
                out = phi[:, :-1, :-1, 0]
        else:
            self.expected_value = [phi[0, -1, -1, i] for i in range(phi.shape[3])]
            if flat_output:
                out = [phi[0, :-1, :-1, i] for i in range(self.model.num_outputs)]
            else:
                out = [phi[:, :-1, :-1, i] for i in range(self.model.num_outputs)]
        return out



    def assert_additivity(self, phi, model_output):

        def check_sum(sum_val, model_output):
            diff = np.abs(sum_val - model_output)
            if np.max(diff / (np.abs(sum_val) + 1e-2)) > 1e-2:
                ind = np.argmax(diff)
                err_msg = "Additivity check failed in TreeExplainer! Please ensure the data matrix you passed to the " \
                          "explainer is the same shape that the model was trained on. If your data shape is correct " \
                          "then please report this on GitHub."
                if self.feature_perturbation != "interventional":
                    err_msg += " Consider retrying with the feature_perturbation='interventional' option."
                err_msg += " This check failed because for one of the samples the sum of the SHAP values" \
                           " was %f, while the model output was %f. If this difference is acceptable" \
                           " you can set check_additivity=False to disable this check." % (sum_val[ind], model_output[ind])
                raise Exception(err_msg)

        if type(phi) is list:
            for i in range(len(phi)):
                check_sum(self.expected_value[i] + phi[i].sum(-1), model_output[:,i])
        else:
            check_sum(self.expected_value + phi.sum(-1), model_output)

    @staticmethod
    def supports_model_with_masker(model, masker):
        """ Determines if this explainer can handle the given model.

        This is an abstract static method meant to be implemented by each subclass.
        """

        if not isinstance(masker, (maskers.Independent)) and masker is not None:
            return False

        try:
            TreeEnsemble(model)
        except:
            return False
        return True
