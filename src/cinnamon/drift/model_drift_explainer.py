import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from .abstract_drift_explainer import AbstractDriftExplainer
from ..model_parser.abstract_model_parser import AbstractModelParser
from .adversarial_drift_explainer import AdversarialDriftExplainer
from ..model_parser.xgboost_parser import XGBoostParser
from ..model_parser.catboost_parser import CatBoostParser
from ..model_parser.base_model_parser import BaseModelParser

from .drift_utils import (compute_drift_num, plot_drift_num,
                          DriftMetricsNum, PerformanceMetricsDrift,
                          compute_model_agnostic_drift_value_num,
                          compute_model_agnostic_drift_value_cat)
from ..common.dev_utils import safe_isinstance
from ..common.stat_utils import (compute_classification_metrics,
                                 compute_regression_metrics)
from ..common.constants import TreeBasedDriftValueType, ModelAgnosticDriftValueType


class ModelDriftExplainer(AbstractDriftExplainer):
    """
    Tool to study data drift between two datasets, in a context where "model" is
    used to make predictions.

    Parameters
    ----------

    model : a XGBoost model (either XGBClassifier, XGBRegressor, XGBRanker, Booster)
        The model used to make predictions.

    iteration_range : Tuple[int, int], optional (default=None)
        Specifies which layer of trees are used. For example, if XGBoost is
        trained with 100 rounds, specifying iteration_range=(10, 20) then only
        the trees built during [10, 20) (half open set) iterations are used.
        If None, all trees are used.

    task : string
        Task corresponding to the (X, Y) data. Either "regression", "classification",
        or "ranking". "task" must be provided if the model is treated as a black box predictor
        (no specific parser for the model).
    
    Attributes
    ----------
    predictions1 : numpy array
        Array of predictions of "model" on X1 (for classification, corresponds
        to raw predictions).

    predictions2 : numpy array
        Array of predictions of "model" on X2 (for classification, corresponds
        to raw predictions).

    pred_proba1 : numpy array
        Array of predicted probabilities of "model" on X1 (equal to None if
        regression or ranking).

    pred_proba2 : numpy array
        Array of predicted probabilities of "model" on X2 (equal to None if
        regression or ranking).

    iteration_range : tuple of integers
        Layer of trees used.

    feature_drifts : list of dict
        Drift measures for each input feature in X.

    target_drift : dict
        Drift measures for the labels y.

    n_features : int
        Number of features in input X.

    feature_names : list of string
        Feature names for input X.

    class_names : list of string
        Class names of the target when task is "classification". Otherwise equal to None.

    cat_feature_indices : list of int
        Indexes of categorical features in input X (not implemented yet: only numerical
        features are allowed currently).

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
        self.iteration_range = self._model_parser.iteration_range
        self.task = self._model_parser.task

        # init other
        self.predictions1 = None
        self.predictions2 = None
        self.pred_proba1 = None
        self.pred_proba2 = None
        self._prediction_dim = None
        self._tree_based_drift_values_sum_check = False

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

        cat_feature_indices: TODO

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

        # fit model parser on data
        self._model_parser.fit(self.X1, self.X2, self.sample_weights1, self.sample_weights2)

        return self

    def get_prediction_drift(self, prediction_type: str = "raw") -> List[DriftMetricsNum]:
        """
        Compute drift measures based on model predictions.

        See the documentation in README for explanations about how it is computed,
        especially the slide presentation.

        Parameters
        ----------
        prediction_type: str, optional (default="raw")
            Type of predictions to consider.
            Choose among:
            - "raw" : logit predictions (binary classification), log-softmax predictions
            (multiclass classification), regular predictions (regression)
            - "proba" : predicted probabilities (only for classification model)

        Returns
        -------
        prediction_drift : list of DriftMetricsNum object
            Drift measures for each predicted dimension.
        """
        pred1, pred2 = self._get_predictions(prediction_type)
        return self._compute_prediction_drift(pred1, pred2, self.task, self._prediction_dim,
                                                self.sample_weights1, self.sample_weights2) 

    @staticmethod
    def _compute_prediction_drift(predictions1, predictions2, task, prediction_dim,
                                  sample_weights1=None, sample_weights2=None) -> List[DriftMetricsNum]:
        prediction_drift = []
        if task == 'classification':
            if prediction_dim == 1:  # binary classif
                prediction_drift.append(compute_drift_num(predictions1, predictions2, sample_weights1, sample_weights2))
            else:  # multiclass classif
                for i in range(predictions1.shape[1]):
                    drift = compute_drift_num(predictions1[:, i], predictions2[:, i],
                                              sample_weights1, sample_weights2)
                    prediction_drift.append(drift)
        elif task in ['regression', 'ranking']:
            prediction_drift.append(compute_drift_num(predictions1, predictions2, sample_weights1, sample_weights2))
        return prediction_drift

    def plot_prediction_drift(self, prediction_type='raw', bins = 10,
                              figsize: Tuple[int, int] = (7, 5),
                              legend_labels: Tuple[str, str] = ('Dataset 1', 'Dataset 2')) -> None:
        """
        Plot histogram of distribution of predictions for dataset 1 and dataset 2
        in order to visualize a potential drift of the predicted values.
        See the documentation in README for explanations about how it is computed,
        especially the slide presentation.

        Parameters
        ----------
        prediction_type: str, optional (default="raw")
            Type of predictions to consider.
            Choose among:
            - "raw" : logit predictions (binary classification), log-softmax predictions
            (multiclass classification), regular predictions (regression)
            - "proba" : predicted probabilities (only for classification models)

        bins : int or sequence of scalars or str, optional (default=10)
            For regression only. 'two_heads' corresponds to a number of bins which is the minimum of
            of the optimal number of bins for dataset 1 and dataset 2 taken separately.
            Other value of "bins" parameter passed to matplotlib.pyplot.hist function are also
            accepted.

        figsize : Tuple[int, int], optional (default=(7, 5))
            Graphic size passed to matplotlib

        legend_labels : Tuple[str, str] (default=('Dataset 1', 'Dataset 2'))
            Legend labels used for dataset 1 and dataset 2

        Returns
        -------
        None
        """
        pred1, pred2 = self._get_predictions(prediction_type)

        if self.task == 'classification' and self._prediction_dim > 1:  # multiclass classif
            for i in range(self._prediction_dim):
                plot_drift_num(pred1[:, i], pred2[:, i], self.sample_weights1, self.sample_weights2,
                               title=f'{self.class_names[i]}', figsize=figsize, bins=bins, legend_labels=legend_labels)
        else:  # binary classif or regression
            plot_drift_num(pred1, pred2, self.sample_weights1, self.sample_weights2, title=f'Predictions',
                           figsize=figsize, bins=bins, legend_labels=legend_labels)

    def get_performance_metrics_drift(self) -> PerformanceMetricsDrift:
        """
        Compute performance metrics on dataset 1 and dataset 2.

        Returns
        -------
        Dictionary of performance metrics
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

    def get_tree_based_drift_values(self, type: str = TreeBasedDriftValueType.NODE_SIZE.value) -> np.array:
        """
        Compute drift values using the tree structures present in the model.

        See the documentation in README for explanations about how it is computed,
        especially the slide presentation.

        Parameters
        ----------
        type: str, optional (default="node_size")
            Method used for drift values computation.
            Choose among:
            - "node_size" (recommended)
            - "mean"
            - "mean_norm"

            See details in slide presentation.

        Returns
        -------
        drift_values : numpy array
        """
        if not self._tree_based_drift_values_sum_check:
            self._model_parser.check_tree_based_drift_values_sum(self.X1, self.X2, self.sample_weights1,
                                                                  self.sample_weights2)
            self._tree_based_drift_values_sum_check = True
        return self._model_parser.compute_tree_based_drift_values(type)

    def plot_tree_based_drift_values(self, n: int = 10, type: str = TreeBasedDriftValueType.NODE_SIZE.value) -> None:
        """
        Plot drift values computed using the tree structures present in the model.

        See the documentation in README for explanations about how it is computed,
        especially the slide presentation.

        Parameters
        ----------
        n : int, optional (default=10)
            Top n features to represent in the plot.

        type: str, optional (default="node_size")
            Method used for drift values computation.
            Choose among:
            - "node_size" (recommended)
            - "mean"
            - "mean_norm"

            See details in slide presentation.

        Returns
        -------
        None
        """
        if self._model_parser is None:
            raise ValueError('You need to run drift_explainer.fit before you can plot drift_values')

        drift_values = self.get_tree_based_drift_values(type=type)
        self._plot_drift_values(drift_values, n, self.feature_names)

    def _get_predictions(self, prediction_type: str) -> Tuple[np.array, np.array]:
        if self.predictions1 is None:
            raise ValueError('You must call the fit method before ploting drift')
        if prediction_type not in ['raw', 'proba']:
            raise ValueError(f'Bad value for prediction_type: {prediction_type}')
        if prediction_type == 'raw':
            pred1, pred2 = self.predictions1, self.predictions2
        else:
            pred1, pred2 = self.pred_proba1, self.pred_proba2
        return pred1, pred2

    def get_model_agnostic_drift_values(self, type: str = ModelAgnosticDriftValueType.MEAN.value, prediction_type: str = "raw",
                                        max_ratio: float = 10, max_n_cat: int = 20) -> np.array:
        if type not in [x.value for x in ModelAgnosticDriftValueType]:
            raise ValueError(f'Bad value for "type": {type}')
        pred1, pred2 = self._get_predictions(prediction_type)
        return self._compute_model_agnostic_drift_values(self.X1, self.X2, type, self.sample_weights1, self.sample_weights2,
                                                           pred1, pred2, self.n_features, self.cat_feature_indices,
                                                           max_ratio, max_n_cat)

    @staticmethod
    def _compute_model_agnostic_drift_values(X1: pd.DataFrame, X2: pd.DataFrame, type: str, sample_weights1: np.array,
                                               sample_weights2: np.array, predictions1: np.array, predictions2: np.array, n_features: int,
                                               cat_feature_indices: List[int], max_ratio: float, max_n_cat: int) -> np.array:

        drift_values = []
        for i in range(n_features):
            if i in cat_feature_indices:
                drift_value = compute_model_agnostic_drift_value_cat(X1.iloc[:, i].values, X2.iloc[:, i].values, type,
                                                                        sample_weights1, sample_weights2, predictions1, predictions2,
                                                                        max_ratio, max_n_cat)
            else:
                drift_value = compute_model_agnostic_drift_value_num(X1.iloc[:, i].values, X2.iloc[:, i].values, type, 
                                                                    sample_weights1, sample_weights2,
                                                                    predictions1, predictions2, bins='two_heads', max_ratio=max_ratio)
            drift_values.append(drift_value)
        return np.array(drift_values)
    
    def plot_model_agnostic_drift_values(self, n: int = 10, type: str = ModelAgnosticDriftValueType.MEAN.value, prediction_type: str = "raw",
                                        max_ratio: float = 10, max_n_cat: int = 20) -> None:
        if type not in [x.value for x in ModelAgnosticDriftValueType]:
            raise ValueError(f'Bad value for "type": {type}')
        drift_values = self.get_model_agnostic_drift_values(type, prediction_type, max_ratio, max_n_cat)
        self._plot_drift_values(drift_values, n, self.feature_names)

    def plot_tree_drift(self, tree_idx: int, type: str = TreeBasedDriftValueType.NODE_SIZE.value) -> None:
        """
        Plot the representation of a given tree in the model, to illustrate how
        drift values are computed.

        See the documentation in README for explanations about how it is computed,
        especially the slide presentation.

        Parameters
        ----------
        tree_idx : int
            Index of the tree to plot

        type: str, optional (default="node_size")
            Method used for drift values computation.
            Choose among:
            - "node_size" (recommended)
            - "mean"
            - "mean_norm"

            See details in slide presentation.

        Returns
        -------
        None
        """
        self._model_parser.plot_tree_drift(tree_idx, type, self.feature_names)

    def get_tree_based_correction_weights(self, max_depth: int = None, max_ratio: int = 10) -> np.array:
        """
        Not recommended way to compute correction weights for data drift (only for
        research purpose). AdversarialDriftExplainer should be preferred for this
        purpose.
        The approach is to use similar ideas as in get_tree_based_drift_values
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
    def _get_class_names(task, model_parser: AbstractModelParser, prediction_dim) -> List[str]:
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
            raise ValueError(f'"cat_feature_indices" argument: {self.cat_feature_indices} not consistent with '
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
                raise ValueError(f'"task" argument must not be empty when model is treated as a black box')
            else:
                self._model_parser: AbstractModelParser = BaseModelParser(model, 'unknown', task)
                ModelDriftExplainer.logger.info(f'The model of type {type(model).__name__} has no specific support in ' 
                                'ModelDriftExplainer. However model agnostic methods only relying on model.predict / model.predict_proba calls are '
                                'available')

        if task and self._model_parser.task and self._model_parser.task != task:
            ModelDriftExplainer.logger.warning(f'task "{task}" passed as parameter not consistent with inferred task for '
                f'model of type {type(model).__name__}. Inferred task {self._model_parser.task} is used')
        if self._model_parser.task == 'ranking':
            ModelDriftExplainer.logger.warning('A ranking model was passed to DriftExplainer. It will be treated similarly as'
                                               ' regression model but there is no warranty about the result')

    @staticmethod
    def _raise_not_implem_feature_error():
        raise NotImplementedError('Model agnostic drift values (only based on model predictions) will be '
                                  'implemented in future version')
