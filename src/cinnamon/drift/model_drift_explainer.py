import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from .abstract_drift_explainer import AbstractDriftExplainer
from ..model_parser.abstract_model_parser import AbstractModelParser
from .adversarial_drift_explainer import AdversarialDriftExplainer
from ..model_parser.xgboost_parser import XGBoostParser
from ..model_parser.catboost_parser import CatBoostParser
from ..model_parser.model_agnostic_model_parser import ModelAgnosticModelParser

from .drift_utils import (compute_drift_cat, compute_drift_num,
                          DriftMetricsNum, PerformanceMetricsDrift,
                          compute_model_agnostic_drift_value_num,
                          compute_model_agnostic_drift_value_cat)
from ..common.dev_utils import safe_isinstance
from ..common.stat_utils import (compute_classification_metrics,
                                 compute_regression_metrics)
from ..common.constants import TreeBasedDriftValueType, ModelAgnosticDriftValueType


class ModelDriftExplainer(AbstractDriftExplainer):
    """
    Study data drift through the lens of a ML model or ML pipeline.
    
    Parameters
    ----------

    model : a ML model or ML pipeline (see "Supported Model" section).
        The model used to make predictions.

    iteration_range : Tuple[int, int], optional (default=None)
        Only for tree based models. 
        Specifies which layer of trees are used. For example, if XGBoost is
        trained with 100 rounds, with iteration_range=(10, 20) then only
        the trees built during [10, 20) iterations are used.
        If None, all trees are used.

    task : string, optional (default=None)
        Task corresponding to the (X, Y) data. Either "regression", "classification",
        or "ranking". "task" is a mandatory parameter if the model is treated as a black box predictor.
    
    Attributes
    ----------
    predictions1 : numpy array
        Array of predictions of "model" on X1 dataset. For classification, corresponds
        to raw (logit of log-softmax) predictions.

    predictions2 : numpy array
        Array of predictions of "model" on X2 dataset. For classification, corresponds
        to raw (logit of log-softmax) predictions.

    pred_proba1 : numpy array
        Array of predicted probabilities of "model" on X1 (equal to None if task is
        regression or ranking).

    pred_proba2 : numpy array
        Array of predicted probabilities of "model" on X2 (equal to None if task is
        regression or ranking).

    iteration_range : tuple of integers
        Layer of trees used.

    feature_drifts : list of Union[DriftMetricsCat, DriftMetricsNum]
        Drift measures for each input feature in X.

    target_drift : Union[DriftMetricsCat, DriftMetricsNum]
        Drift measures for the labels y.

    n_features : int
        Number of features in input X.

    feature_names : list of string
        Feature names for input X.

    class_names : list of string
        Class names of the target when task is "classification". Otherwise equal to None.

    cat_feature_indices : list of int
        Indexes of categorical features in input X.

    X1, X2 : pandas dataframes
        X1 and X2 inputs passed to the "fit" method.

    y1, y2 : numpy arrays
        y1 and y2 targets passed to the "fit" method.

    sample_weights1, sample_weights2 : numpy arrays
        sample_weights1 and sample_weights2 arrays passed to the "fit" method.
    """

    def __init__(self, model, iteration_range: Tuple[int, int] = None, task: Optional[str] = None):
        super().__init__()
        # Parse model
        self._parse_model(model, iteration_range, task)
        self.task = self._model_parser.task

        # init other
        self.predictions1 = None
        self.predictions2 = None
        self.pred_proba1 = None
        self.pred_proba2 = None
        self._prediction_dim = None
        self._tree_based_drift_importances_sum_check = False

    def fit(self, X1: pd.DataFrame, X2: pd.DataFrame, y1: np.array=None, y2: np.array= None,
            sample_weights1: np.array = None, sample_weights2: np.array = None,
            cat_feature_indices: Optional[List[int]] = None):
        """
        Fit the model drift explainer to dataset 1 and dataset 2.

        Parameters
        ----------
        X1 : pandas dataframe of shape (n_samples, n_features)
            Dataset 1 inputs.

        X2 : pandas dataframe of shape (n_samples, n_features)
            Dataset 2 inputs.

        y1 : numpy array of shape (n_samples,), optional (default=None)
            Dataset 1 labels.
            If None, data drift is only analyzed based on inputs X1 and X2

        y2 : numpy array of shape (n_samples,), optional (default=None)
            Dataset 2 labels.
            If None, data drift is only analyzed based on inputs X1 and X2

        sample_weights1: numpy array of shape (n_samples,), optional (default=None)
            Array of weights that are assigned to individual samples of dataset 1
            If None, then each sample of dataset 1 is given unit weight.

        sample_weights2: numpy array of shape (n_samples,), optional (default=None)
            Array of weights that are assigned to individual samples of dataset 2
            If None, then each sample of dataset 2 is given unit weight.

        cat_feature_indices: list of int
        Indexes of categorical features in input X.

        Returns
        -------
        ModelDriftExplainer
            The fitted model drift explainer.
        """
        # Check arguments and save them as attributes
        self._check_fit_arguments(X1, X2, y1, y2, sample_weights1, sample_weights2, cat_feature_indices)

        # check coherence between parsed model and fit arguments
        self._check_coherence()

        # set some class attributes
        self.n_features = self._get_n_features(self._model_parser, self.X1)
        self.feature_names = self._get_feature_names(self.X1, self.X2, self._model_parser)
        self.cat_feature_indices = self._get_cat_feature_indices(self.cat_feature_indices)
        self._prediction_dim = self._model_parser.get_prediction_dim(self.X1)
        self.class_names = self._get_class_names(self.task, self._model_parser, self._prediction_dim)


        # compute model predictions
        self.predictions1 = self._model_parser.get_predictions(self.X1, prediction_type="raw")
        self.predictions2 = self._model_parser.get_predictions(self.X2, prediction_type="raw")
        if self.task == "classification":
            self.pred_proba1 = self._model_parser.get_predictions(self.X1, prediction_type="proba")
            self.pred_proba2 = self._model_parser.get_predictions(self.X2, prediction_type="proba")
            self.pred_class1 = self._model_parser.get_predictions(self.X1, prediction_type="class")
            self.pred_class2 = self._model_parser.get_predictions(self.X2, prediction_type="class")

        # fit model parser on data
        self._model_parser.fit(self.X1, self.X2, self.sample_weights1, self.sample_weights2)

        return self

    def get_prediction_drift(self, prediction_type: str = "raw") -> List[DriftMetricsNum]:
        """
        Compute drift measures based on model predictions.

        Parameters
        ----------
        prediction_type: str, optional (default="raw")
            Type of predictions to consider.
            Choose among:
            - "raw" : logit predictions (binary classification), log-softmax predictions
            (multiclass classification), regular predictions (regression)
            - "proba" : predicted probabilities (only for classification model)
            - "class": predicted classes (only for classification model)

        Returns
        -------
        prediction_drift : list of DriftMetricsNum or DriftMetricsCat objects
            Drift measures for each predicted dimension.
        """
        pred1, pred2 = self._get_predictions(prediction_type)
        return self._compute_prediction_drift(pred1, pred2, prediction_type, self.task, self._prediction_dim,
                                                self.sample_weights1, self.sample_weights2) 

    @staticmethod
    def _compute_prediction_drift(predictions1, predictions2, prediction_type, task, prediction_dim,
                                  sample_weights1=None, sample_weights2=None) -> List[DriftMetricsNum]:
        prediction_drift = []
        if task == 'classification':
            if prediction_type == 'class':
                prediction_drift.append(compute_drift_cat(predictions1, predictions2, sample_weights1, sample_weights2))
            else: # prediction_type in ['raw', 'proba']
                if prediction_dim == 1:  # binary classif
                    prediction_drift.append(compute_drift_num(predictions1, predictions2, sample_weights1, sample_weights2))
                else:  # multiclass classif
                    for i in range(predictions1.shape[1]):
                        drift = compute_drift_num(predictions1[:, i], predictions2[:, i],
                                                sample_weights1, sample_weights2)
                        prediction_drift.append(drift)
        else: # task in ['regression', 'ranking']:
            prediction_drift.append(compute_drift_num(predictions1, predictions2, sample_weights1, sample_weights2))
        return prediction_drift

    def get_performance_metrics_drift(self) -> PerformanceMetricsDrift:
        """
        Compute performance metrics on dataset 1 and dataset 2.

        Returns
        -------
        performance_metrics_drift: PerformanceMetricsDrift object
            Comparison of either RegressionMetrics or ClassificaionMetrics objects.
        """
        if self.y1 is None or self.y2 is None:
            self._raise_no_target_error()
        if self.task == 'classification':
            return PerformanceMetricsDrift(dataset1=compute_classification_metrics(self.y1, self.pred_proba1,
                                                                                   self.sample_weights1, self.class_names),
                                           dataset2=compute_classification_metrics(self.y2, self.pred_proba2,
                                                                                   self.sample_weights2, self.class_names))
        elif self.task == 'regression':
            return PerformanceMetricsDrift(dataset1=compute_regression_metrics(self.y1, self.predictions1,
                                                                               self.sample_weights1),
                                           dataset2=compute_regression_metrics(self.y2, self.predictions2,
                                                                               self.sample_weights2))
        else:  # ranking
            raise NotImplementedError('No metrics currently implemented for ranking model')

    def get_tree_based_drift_importances(self, type: str = TreeBasedDriftValueType.MEAN.value) -> np.array:
        """
        Compute drift importances using the tree structure of the model.

        See the documentation in README for explanations about how it is computed,
        especially the slide presentation.

        Parameters
        ----------
        type: str, optional (default="mean")
            Method used for drift importances computation.
            Choose among:
            - "node_size"
            - "mean"
            - "mean_norm"

            See details in slide presentation.

        Returns
        -------
        drift_importances : numpy array
        """
        if not self._tree_based_drift_importances_sum_check:
            self._model_parser.check_tree_based_drift_importances_sum(self.X1, self.X2, self.sample_weights1,
                                                                  self.sample_weights2)
            self._tree_based_drift_importances_sum_check = True
        return self._model_parser.compute_tree_based_drift_importances(type)

    def _get_predictions(self, prediction_type: str) -> Tuple[np.array, np.array]:
        if self.predictions1 is None:
            raise ValueError('You must call the fit method before accessing the predictions')
        if prediction_type not in ['raw', 'proba', 'class']:
            raise ValueError(f'Bad value for prediction_type: {prediction_type}')
        if prediction_type == 'raw':
            pred1, pred2 = self.predictions1, self.predictions2
        elif prediction_type == 'proba':
            pred1, pred2 = self.pred_proba1, self.pred_proba2
        else:
            pred1, pred2 = self.pred_class1, self.pred_class2
        return pred1, pred2

    def get_model_agnostic_drift_importances(self, type: str = ModelAgnosticDriftValueType.MEAN.value, prediction_type: str = "raw",
                                        max_ratio: float = 10, max_n_cat: int = 20) -> np.array:
        """
        Compute drift importances using the model agnostic method.

        See the documentation in README for explanations about how it is computed,
        especially the slide presentation.

        Parameters
        ----------
        type: str, optional (default="mean")
            Method used for drift importances computation.
            Choose among:
            - "mean"
            - "wasserstein"

            See details in slide presentation.

        prediction_type: str,  optional (default="raw")
            Choose among:
            - "raw"
            - "proba": predicted probability if task == 'classification'
            - "class": predicted class if task == 'classification'

        max_ratio: int, optional (default=10)
            Only used for categorical features

        max_n_cat: int, optional (default=20)
            Only used for categorical features
        
        Returns
        -------
        drift_importances : numpy array
        """
        if type not in [x.value for x in ModelAgnosticDriftValueType]:
            raise ValueError(f'Bad value for "type": {type}')
        pred1, pred2 = self._get_predictions(prediction_type)
        return self._compute_model_agnostic_drift_importances(self.X1, self.X2, type, self.sample_weights1, self.sample_weights2,
                                                           pred1, pred2, self.n_features, self.cat_feature_indices,
                                                           max_ratio, max_n_cat)

    @staticmethod
    def _compute_model_agnostic_drift_importances(X1: pd.DataFrame, X2: pd.DataFrame, type: str, sample_weights1: np.array,
                                               sample_weights2: np.array, predictions1: np.array, predictions2: np.array, n_features: int,
                                               cat_feature_indices: List[int], max_ratio: float, max_n_cat: int) -> np.array:

        drift_importances = []
        for i in range(n_features):
            if i in cat_feature_indices:
                drift_value = compute_model_agnostic_drift_value_cat(X1.iloc[:, i].values, X2.iloc[:, i].values, type,
                                                                        sample_weights1, sample_weights2, predictions1, predictions2,
                                                                        max_ratio, max_n_cat)
            else:
                drift_value = compute_model_agnostic_drift_value_num(X1.iloc[:, i].values, X2.iloc[:, i].values, type, 
                                                                    sample_weights1, sample_weights2,
                                                                    predictions1, predictions2, bins='two_heads', max_ratio=max_ratio)
            drift_importances.append(drift_value)
        return np.array(drift_importances)
    
    def get_tree_based_correction_weights(self, max_depth: int = None, max_ratio: int = 10) -> np.array:
        """
        Not recommended way to compute correction weights for data drift (only for
        research purpose). AdversarialDriftExplainer should be preferred for this
        purpose.
        The approach is to use similar ideas as in get_tree_based_drift_importances
        in order to estimate correction weights (but first experiments show it has
        bad performance).

        Parameters
        ----------
        max_depth : int, optional (default=None)
            Depth at which the ratio of node weights are computed
            If None, ratio are computed in terminal leaves

        max_ratio: int, optional (default=10)
            Maximum ratio between two weights returned in correction_weights (weights
            are thresholded so that the ratio between two weights do not exceed
            max_ratio)

        Returns
        -------
        correction_weights : np.array
            Array of correction weights for the samples of dataset 1
        """
        return self._model_parser.compute_tree_based_correction_weights(self.X1, max_depth, max_ratio, self.sample_weights1)

    @staticmethod
    def _get_n_features(model_parser: AbstractModelParser, X1) -> int:
        if model_parser.n_features:
            return model_parser.n_features
        else:
            return X1.shape[1]

    @staticmethod
    def _get_feature_names(X1: pd.DataFrame, X2: pd.DataFrame, model_parser: AbstractModelParser):
        # we take feature names in X1 and X2 column names if provided
        if list(X1.columns) != list(X2.columns):
            raise ValueError('"X1.columns" and "X2.columns" are not equal')
        feature_names = list(X1.columns)

        # if catboost model, check that order of columns in X1 is consistent with feature names in catboost
        if model_parser.model_type in ['catboost.core.CatBoostClassifier', 'catboost.core.CatBoostRegressor']:
            if model_parser.feature_names is not None and model_parser.feature_names != feature_names:
                raise ValueError('X1.columns and CatBoost "feature_names_" are ot equal')

        return feature_names

    @staticmethod
    def _get_cat_feature_indices(cat_feature_indices: Optional[List[int]]):
        if cat_feature_indices:
            return cat_feature_indices
        else:
            return []

    @staticmethod
    def _get_class_names(task, model_parser: AbstractModelParser, prediction_dim: int) -> List[str]:
        if task == 'regression':
            return []
        elif model_parser.model_type == 'catboost.core.CatBoostClassifier':
            return model_parser.class_names
        else:
            n_class = 2 if prediction_dim == 1 else prediction_dim
            return [str(i) for i in range(n_class)]

    def _check_coherence(self):
        if self._model_parser.n_features and self._model_parser.n_features != self.X1.shape[1]:
            raise ValueError('Number of columns in X1 (X2) not equal to the number of features required for "model"')
        if self._model_parser.cat_feature_indices and self._model_parser.cat_feature_indices != self.cat_feature_indices:
            ModelDriftExplainer.logger.warning(f'"cat_feature_indices" argument: {self.cat_feature_indices} not consistent with ' +
                            f'value inferred from the model: {self._model_parser.cat_feature_indices}')

    def _parse_model(self, model, iteration_range: Optional[Tuple[int, int]], task: Optional[str]):
        if safe_isinstance(model, 'xgboost.core.Booster'):
            self._model_parser: AbstractModelParser = XGBoostParser(model, 'xgboost.core.Booster', iteration_range)
        elif safe_isinstance(model, 'xgboost.sklearn.XGBClassifier'):
            # output of get_booster() is in binary format and universal among various XGBoost interfaces
            self._model_parser: AbstractModelParser = XGBoostParser(model.get_booster(), 'xgboost.sklearn.XGBClassifier',
                                                                   iteration_range)
        elif safe_isinstance(model, 'xgboost.sklearn.XGBRegressor'):
            self._model_parser: AbstractModelParser = XGBoostParser(model.get_booster(), 'xgboost.sklearn.XGBRegressor',
                                                                   iteration_range)
        elif safe_isinstance(model, 'xgboost.sklearn.XGBRanker'):
            self._model_parser: AbstractModelParser = XGBoostParser(model.get_booster(), 'xgboost.sklearn.XGBRanker',
                                                            iteration_range)
        # TODO: because of unresolved pbms with CatBoost().calc_leaf_indexes(), CatBoost is not supported in CinnaMon
        elif safe_isinstance(model, 'catboost.core.CatBoostClassifier'):
            self._model_parser: AbstractModelParser = CatBoostParser(model, 'catboost.core.CatBoostClassifier',
                                                             iteration_range, task='classification')
        elif safe_isinstance(model, 'catboost.core.CatBoostRegressor'):
            self._model_parser: AbstractModelParser = CatBoostParser(model, 'catboost.core.CatBoostRegressor',
                                                              iteration_range, task='regression')
        else:
            if not task:
                raise ValueError(f'Model of type {type(model).__name__} has no specific support in CinnaMon ModelDriftExplainer. '
                                '"task" argument should be provided so that model can be processed as a black box and model '
                                'agnostic features are available.')
            else:
                self._model_parser: AbstractModelParser = ModelAgnosticModelParser(model, 'unknown', task)
                #ModelDriftExplainer.logger.info(f'Model of type {type(model).__name__} has no specific support in ' 
                #                'ModelDriftExplainer. However model agnostic methods only relying on model.predict / model.predict_proba calls are '
                #                'available')

        if task and self._model_parser.task and self._model_parser.task != task:
            ModelDriftExplainer.logger.warning(f'task "{task}" passed as parameter not consistent with inferred task for '
                f'model of type {type(model).__name__}. Inferred task {self._model_parser.task} is used')
        if self._model_parser.task == 'ranking':
            ModelDriftExplainer.logger.warning('A ranking model was passed to DriftExplainer. It will be treated similarly as'
                                               ' regression model but there is no warranty about the result')

    @staticmethod
    def _raise_not_implem_feature_error():
        raise NotImplementedError('Model agnostic drift importances (only based on model predictions) will be '
                                  'implemented in future version')
