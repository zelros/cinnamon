import logging
import numpy as np
import pandas as pd
from typing import List, Tuple

from .drift_explainer_abc import DriftExplainerABC
from ..model_parser.i_model_parser import IModelParser
from .adversarial_drift_explainer import AdversarialDriftExplainer
from ..model_parser.xgboost_parser import XGBoostParser

from .drift_utils import (compute_drift_num, plot_drift_num)
from ..common.dev_utils import safe_isinstance
from ..common.math_utils import compute_classification_metrics, compute_regression_metrics


class ModelDriftExplainer(DriftExplainerABC):

    logger = logging.getLogger('DriftExplainer')

    def __init__(self, model, iteration_range: Tuple[int,int] = None):
        super().__init__()
        # Parse model
        self.iteration_range = iteration_range  # just because it is param passed by user (no it can be modified...)
        self._parse_model(model, self.iteration_range)
        self.task = self.model_parser.task

        # init other
        self.prediction_dim = None
        self.predictions1 = None
        self.predictions2 = None
        self.pred_proba1 = None
        self.pred_proba2 = None

    def fit(self, X1: pd.DataFrame, X2: pd.DataFrame, y1: np.array=None, y2: np.array= None,
            sample_weights1: np.array = None, sample_weights2: np.array = None):
        """
        Compute drift coefficients by feature. For algebraic drift, we do X2 - X1
        :param model: the (tree based) model we want to analyze the drift on. In binary format.
        :param X1: the base X in the comparison (usually the train X or validation X)
        :param X2: the X which is compared with X1 (usually the test X of production X)
        :return:
        """

        # Check arguments and save them as attributes
        self._check_fit_arguments(X1, X2, y1, y2, sample_weights1, sample_weights2)

        # check coherence between parsed model and fit arguments
        self._check_coherence()

        # set some class attributes
        self.n_features = self._get_n_features(self.model_parser)
        self.feature_names = self._get_feature_names(self.X1, self.X2, self.model_parser)
        self.cat_feature_indices = self._get_cat_feature_indices(self.model_parser)
        self.class_names = self._get_class_names(self.task, self.model_parser)
        self.prediction_dim = self.model_parser.prediction_dim

        # compute model predictions
        self.predictions1 = self.model_parser.get_predictions(self.X1, prediction_type="raw")
        self.predictions2 = self.model_parser.get_predictions(self.X2, prediction_type="raw")
        if self.task == "classification":
            self.pred_proba1 = self.model_parser.get_predictions(self.X1, prediction_type="proba")
            self.pred_proba2 = self.model_parser.get_predictions(self.X2, prediction_type="proba")

        # fit model parser on data
        self.model_parser.fit(self.X1, self.X2, self.sample_weights1, self.sample_weights2)

        return self

    def get_prediction_drift(self, prediction_type="raw"):
        if prediction_type not in ['raw', 'proba']:
            raise ValueError(f'Bad value for prediction_type: {prediction_type}')
        if prediction_type == 'raw':
            return self._compute_prediction_drift(self.predictions1, self.predictions2, self.task, self.prediction_dim,
                                                  self.sample_weights1, self.sample_weights2)
        elif prediction_type == 'proba':
            return self._compute_prediction_drift(self.pred_proba1, self.pred_proba2, self.task, self.prediction_dim,
                                                  self.sample_weights1, self.sample_weights2)

    @staticmethod
    def _compute_prediction_drift(predictions1, predictions2, task, prediction_dim, sample_weights1=None, sample_weights2=None):
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

    def plot_prediction_drift(self, prediction_type='raw', figsize=(7, 5)):
        if self.predictions1 is None:
            raise ValueError('You must call the fit method before ploting drift')
        if prediction_type not in ['raw', 'proba']:
            raise ValueError(f'Bad value for prediction_type: {prediction_type}')
        if prediction_type == 'raw':
            pred1, pred2 = self.predictions1, self.predictions2
        else:
            pred1, pred2 = self.pred_proba1, self.pred_proba2

        if self.task == 'classification' and self.model_parser.prediction_dim > 1:  # multiclass classif
            for i in range(self.model_parser.prediction_dim):
                plot_drift_num(pred1[:, i], pred2[:, i], self.sample_weights1, self.sample_weights2,
                               title=f'{self.class_names[i]}', figsize=figsize)
        else:  # binary classif or regression
            plot_drift_num(pred1, pred2, self.sample_weights1, self.sample_weights2, title=f'Predictions',
                           figsize=figsize)

    def get_performance_metrics_drift(self):
        if self.y1 is None or self.y2 is None:
            self._raise_no_target_error()
        if self.task == 'classification':
            return {'dataset 1': compute_classification_metrics(self.y1, self.pred_proba1, self.sample_weights1),
                    'dataset 2': compute_classification_metrics(self.y2, self.pred_proba2, self.sample_weights2)}
        elif self.task == 'regression':
            return {'dataset 1': compute_regression_metrics(self.y1, self.predictions1, self.sample_weights1),
                    'dataset 2': compute_regression_metrics(self.y2, self.predictions2, self.sample_weights2)}
        else:  # ranking
            raise NotImplementedError('No metrics currently implemented for ranking model')

    def get_tree_based_drift_values(self, type: str = 'node_size'):
        return self.model_parser.compute_tree_based_drift_values(type)

    def plot_tree_based_drift_values(self, n: int = 10, type: str = 'node_size'):
        if self.model_parser is None:
            raise ValueError('You need to run drift_explainer.fit before you can plot drift_values')

        drift_values = self.get_tree_based_drift_values(type=type)
        self._plot_drift_values(drift_values, n, self.feature_names)

    def get_prediction_based_drift_values(self) -> np.array:
        self._raise_not_implem_feature_error()
        return self._compute_prediction_based_drift_values(self.X1, self.X2, self.sample_weights1, self.sample_weights2,
                                                           self.predictions1, self.n_features, self.cat_feature_indices,
                                                           self.feature_names)

    def _compute_prediction_based_drift_values(self, X1: pd.DataFrame, X2: pd.DataFrame, sample_weights1: np.array,
                                               sample_weights2: np.array, predictions1: np.array, n_features: int,
                                               cat_feature_indices: List[int], feature_names: List[str]) -> np.array:
        drift_values = []
        for i in range(n_features):
            print(i, feature_names[i])
            if i in cat_feature_indices:
                raise NotImplementedError
                # drift_value = compute_prediction_based_drift_value_cat()
            else:
                drift_value = compute_prediction_based_drift_value_num(X1.iloc[:, i].values, X2.iloc[:, i].values,
                                                                       sample_weights1, sample_weights2, predictions1)
            drift_values.append(drift_value)
        return np.array(drift_values).reshape(-1, 1)

    def plot_prediction_based_drift_values(self, n: int = 10):
        self._raise_not_implem_feature_error()
        drift_values = self.get_prediction_based_drift_values()
        print(drift_values)
        self._plot_drift_values(drift_values, n, self.feature_names)

    def plot_tree_drift(self, tree_idx: int, type: str = 'node_size'):
        return self.model_parser.plot_tree_drift(tree_idx, type, self.feature_names)

    def get_tree_based_correction_weights(self, max_depth=None, max_ratio=10):
        return self.model_parser.compute_tree_based_correction_weights(self.X1, max_depth, max_ratio, self.sample_weights1)

    @staticmethod
    def _get_n_features(model_parser: IModelParser):
        return model_parser.n_features

    @staticmethod
    def _get_feature_names(X1: pd.DataFrame, X2: pd.DataFrame, model_parser: IModelParser):
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
    def _get_cat_feature_indices(model_parser: IModelParser):
        if model_parser.model_type in ['catboost.core.CatBoostClassifier', 'catboost.core.CatBoostRegressor']:
            return model_parser.cat_feature_indices
        else:
            return []  # TODO: maybe add binary features to cat_feature_indices

    @staticmethod
    def _get_class_names(task, model_parser: IModelParser) -> List[str]:
        if task == 'regression':
            return []
        elif model_parser.model_type == 'catboost.core.CatBoostClassifier':
            return model_parser.class_names
        else:
            n_class = 2 if model_parser.prediction_dim == 1 else model_parser.prediction_dim
            return [str(i) for i in range(n_class)]

    def _check_coherence(self):
        if self.model_parser.n_features != self.X1.shape[1]:
            raise ValueError('Number of columns in X1 (X2) not equal to the number of features required for "model"')

    def _parse_model(self, model, iteration_range):
        if safe_isinstance(model, 'xgboost.core.Booster'):
            self.model_parser: IModelParser = XGBoostParser(model, 'xgboost.core.Booster', iteration_range)
        elif safe_isinstance(model, 'xgboost.sklearn.XGBClassifier'):
            # output of get_booster() is in binary format and universal among various XGBoost interfaces
            self.model_parser: IModelParser = XGBoostParser(model.get_booster(), 'xgboost.sklearn.XGBClassifier',
                                                                   iteration_range)
        elif safe_isinstance(model, 'xgboost.sklearn.XGBRegressor'):
            self.model_parser: IModelParser = XGBoostParser(model.get_booster(), 'xgboost.sklearn.XGBRegressor',
                                                                   iteration_range)
        elif safe_isinstance(model, 'xgboost.sklearn.XGBRanker'):
            self.model_parser: IModelParser = XGBoostParser(model.get_booster(), 'xgboost.sklearn.XGBRanker',
                                                            iteration_range)
# because of unresolved pbms with CatBoost().calc_leaf_indexes(), CatBoost is not supported in CinnaMon
#        elif safe_isinstance(model, 'catboost.core.CatBoostClassifier'):
#            self.model_parser: IModelParser = CatBoostParser(model, 'catboost.core.CatBoostClassifier',
#                                                             iteration_range, task='classification')
        else:
            raise TypeError(f'The type of model {type(model).__name__} is not supported in ModelDriftExplainer. Only '
                            f'XGBoost is supported currently. Support for other tree based algorithms and model '
                            f'agnostic methods only relying on model.predict are under development')
            #  model agnostic methods only relying on model.predict are under development
            #  self.model_parser: IModelParser = UnknownModelParser(model, 'unknown')
        if self.model_parser.task == 'ranking':
            ModelDriftExplainer.logger.warning('A ranking model was passed to DriftExplainer. It will be treated similarly as'
                                               ' regression model but there is no warranty about the result')

    @staticmethod
    def _raise_not_implem_feature_error():
        raise NotImplementedError('Model agnostic drift values (only based on model predictions) will be '
                                  'implemented in future version')


def compute_prediction_based_drift_value_num(a1: np.array, a2: np.array, sample_weights1: np.array,
                                             sample_weights2: np.array, predictions1: np.array):
    correction_weights = (AdversarialDriftExplainer(seed=2021, verbosity=True, learning_rate=0.2, tree_method='hist')
                          .fit(a1, a2, sample_weights1=sample_weights1, sample_weights2=sample_weights2)
                          .get_adversarial_correction_weights(max_ratio=10))
    # wasserstein distance between distrib of prediction 1 with original weights, and corrected weights
    drift_value = wasserstein_distance(predictions1, predictions1, correction_weights, sample_weights1)
    plot_drift_num(predictions1, predictions1, correction_weights, sample_weights1)
    return drift_value


def compute_prediction_based_drift_value_cat():
    pass
