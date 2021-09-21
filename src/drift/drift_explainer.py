import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, ks_2samp
from typing import List, Tuple

from .i_drift_explainer import IDriftExplainer
from ..model_parser.i_tree_ensemble import ITreeEnsembleParser
from ..model_parser.catboost import CatBoostParser
from ..model_parser.xgboost import XGBoostParser

from ..drift_correction.i_drift_corrector import IDriftCorrector
from ..drift_correction.adversarial_drift_corrector import AdversarialDriftCorrector
from ..drift_correction.tree_ensemble_drift_corrector import TreeEnsembleDriftCorrector
from ..drift_correction.feature_based_drift_corrector import FeatureBasedDriftCorrector

from .utils import wasserstein_distance_for_cat, compute_distribution_cat, compute_mean_diff, safe_isinstance
from ..report.drift_report_generator import DriftReportGenerator

logging.basicConfig(format='%(levelname)s:%(asctime)s - (%(pathname)s) %(message)s', level=logging.INFO)


def compute_drift_num(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None):
    # TODO: does ks generalize to weighted samples ?
    if (sample_weights1 is None and sample_weights2 is None or
            np.all(sample_weights1 == sample_weights1[0]) and np.all(sample_weights2 == sample_weights2[0])):
        kolmogorov_smirnov = ks_2samp(a1, a2)
    else:
        kolmogorov_smirnov = None
    return {'mean_difference': compute_mean_diff(a1, a2, sample_weights1, sample_weights2),
            'wasserstein': wasserstein_distance(a1, a2, sample_weights1, sample_weights2),
            'kolmogorov_smirnov': kolmogorov_smirnov}


def compute_drift_cat(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None):
    if (sample_weights1 is None and sample_weights2 is None or
            np.all(sample_weights1 == sample_weights1[0]) and np.all(sample_weights2 == sample_weights2[0])):
        # TODO: does chi2 generalize to weighted samples ?
        # indeed, I am sure it is not sufficient to compute the contingency table with weights. So the chi2 formula need
        # to take weights into account
        # chi2 should not take max_n_cat into account. If pbm with number of cat, should be handled by
        # chi2_test internally with a proper solution

        # TODO chi2 not working for now
        #chi2 = chi2_test(np.concatenate((a1, a2)), np.array([0] * len(a1) + [1] * len(a2)))
        chi2 = None
    else:
        chi2 = None
    return {'wasserstein': wasserstein_distance_for_cat(a1, a2, sample_weights1, sample_weights2),
            'chi2_test': chi2}


def plot_drift_cat(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None, title=None,
                   max_n_cat: float = None):

    # compute both distributions
    distrib = compute_distribution_cat(a1, a2, sample_weights1, sample_weights2, max_n_cat)
    bar_height = np.array([v for v in distrib.values()]) # len(distrib) rows and 2 columns

    #plot
    index = np.arange(len(distrib))
    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(index, bar_height[:, 0], bar_width, label="Dataset 1")
    ax.bar(index+bar_width, bar_height[:, 1], bar_width, label="Dataset 2")

    ax.set_xlabel('Category')
    ax.set_ylabel('Percentage')
    ax.set_title(title)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(list(distrib.keys()), rotation=30)
    ax.legend()
    plt.show()


def plot_drift_num(a1: np.array, a2: np.array, sample_weights1: np.array=None, sample_weights2: np.array=None,
                   title=None):
    #distrib = compute_distribution_num(a1, a2, sample_weights1, sample_weights2)
    plt.hist(a1, bins=100, density=True, weights=sample_weights1, alpha=0.3)
    plt.hist(a2, bins=100, density=True, weights=sample_weights2, alpha=0.3)
    plt.legend(['Dataset 1', 'Dataset 2'])
    plt.title(title)
    plt.show()


def compute_distribution_num(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None):
    pass


class DriftExplainer(IDriftExplainer):

    logger = logging.getLogger('DriftExplainer')

    def __init__(self):
        self.model_parser = None
        self.feature_drifts = None
        self.target_drift = None
        #self.prediction_drift = None
        #self.feature_contribs = None
        self.feature_names = None
        self.class_names = None
        self.cat_feature_indices = None  # indexes of categorical_features
        self.n_features = None
        self.prediction_dim = None
        self.predictions1 = None
        self.predictions2 = None
        self.pred_proba1 = None
        self.pred_proba2 = None
        self.task = None
        self.iteration_range = None
        #self.cat_feature_distribs = None

        # usefull for the report when I export the dataset (may not be usefull if I stock only the necessary info
        # to make the plot
        self.X1 = None  # same
        self.X2 = None  # same
        self.sample_weights1 = None  # same
        self.sample_weights2 = None  # same
        self.y1 = None  # same
        self.y2 = None  # same

    def fit(self, model, X1: pd.DataFrame, X2: pd.DataFrame, y1: np.array=None, y2: np.array= None,
            sample_weights1: np.array = None, sample_weights2: np.array = None,
            iteration_range: Tuple[int,int] = None):
        """
        Compute drift coefficients by feature. For algebraic drift, we do X2 - X1
        :param model: the (tree based) model we want to analyze the drift on. In binary format.
        :param X1: the base X in the comparison (usually the train X or validation X)
        :param X2: the X which is compared with X1 (usually the test X of production X)
        :return:
        """

        # Check arguments
        self.sample_weights1 = self._check_sample_weights(sample_weights1, X1)
        self.sample_weights2 = self._check_sample_weights(sample_weights2, X2)
        self._check_X_shape(X1, X2)
        self.X1 = X1
        self.X2 = X2
        self.y1 = y1
        self.y2 = y2

        # Parse model
        DriftExplainer.logger.info('Step 1 - Parse model')
        self.iteration_range = iteration_range  # just because it is param passed by user (no it can be modified...)
        self._parse_model(model, self.iteration_range, X1)
        self.task = self.model_parser.task
        if self.model_parser.n_features != self.X1.shape[1]:
            raise ValueError('Number of columns in X1 (X2) not equal to the number of features required for "model"')

        # Set some class attributes
        self.n_features = self.model_parser.n_features
        self.feature_names = self._get_feature_names(self.X1, self.X2, self.model_parser)
        self.cat_feature_indices = self._get_cat_feature_indices(self.model_parser)
        self.class_names = self._get_class_names(self.task, self.model_parser)
        self.prediction_dim = self.model_parser.prediction_dim

        # Compute predictions
        self.predictions1 = self.model_parser.get_predictions(self.X1, prediction_type="raw")
        self.predictions2 = self.model_parser.get_predictions(self.X2, prediction_type="raw")
        if self.task == "classification":
            self.pred_proba1 = self.model_parser.get_predictions(self.X1, prediction_type="proba")
            self.pred_proba2 = self.model_parser.get_predictions(self.X2, prediction_type="proba")

        # Compute node weights: weighted sum of observations in each node
        self.model_parser.fit(self.X1, self.X2, self.sample_weights1, self.sample_weights2)

        # Drift of each feature of the model
        self.feature_drifts = self._compute_feature_drifts(self.X1, self.X2, self.n_features, self.cat_feature_indices,
                                                           self.sample_weights1, self.sample_weights2)

        # Drift of the target ground truth labels
        self.target_drift = self._compute_target_drift(self.y1, self.y2, self.task, self.sample_weights1,
                                                       self.sample_weights2)

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
        DriftExplainer.logger.info('Evaluate prediction drift')
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

    @staticmethod
    def _compute_feature_drifts(X1, X2, n_features, cat_feature_indices, sample_weights1, sample_weights2):
        DriftExplainer.logger.info('Evaluate univariate drift of each feature')
        feature_drifts = []
        for i in range(n_features):
            if i in cat_feature_indices:
                feature_drift = compute_drift_cat(X1.iloc[:,i].values, X2.iloc[:,i].values,
                                                  sample_weights1, sample_weights2)
            else:
                feature_drift = compute_drift_num(X1.iloc[:,i].values, X2.iloc[:,i].values,
                                                  sample_weights1, sample_weights2)
            feature_drifts.append(feature_drift)
        return feature_drifts

    def get_target_drift(self):
        return self.target_drift

    @staticmethod
    def _compute_target_drift(y1, y2, task, sample_weights1, sample_weights2):
        if y1 is not None and y2 is not None:
            DriftExplainer.logger.info('Evaluate drift of the target ground truth labels')
            if task == 'classification':
                return compute_drift_cat(y1, y2, sample_weights1, sample_weights2)
            elif task in ['regression', 'ranking']:
                return compute_drift_num(y1, y2, sample_weights1, sample_weights2)

    def get_feature_contribs(self, type: str = 'node_size'):
        return self.model_parser.compute_feature_contribs(type)

    def plot_target_drift(self, max_n_cat: int = 20):
        if self.y1 is None or self.y2 is None:
            raise ValueError('"y1" or "y2" argument was not passed to drift_explainer.fit method')
        if self.task == 'classification':
            plot_drift_cat(self.y1, self.y2, self.sample_weights1, self.sample_weights2, title='target',
                           max_n_cat=max_n_cat)
        elif self.task == 'regression':
            plot_drift_num(self.y1, self.y2, self.sample_weights1, self.sample_weights2, title='target')

    def plot_prediction_drift(self, prediction_type='raw'):
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
                               title=f'{self.class_names[i]}')
        else:  # binary classif or regression
            plot_drift_num(pred1, pred2, self.sample_weights1, self.sample_weights2, title=f'Predictions')

    def get_feature_drift(self, feature) -> dict:
        if self.X1 is None:
            raise ValueError('You must call the fit method before calling "get_feature_drift"')
        feature_index, feature_name = self._check_feature_param(feature, self.feature_names)
        return self.feature_drifts[feature_index]

    def get_feature_drifts(self) -> List[dict]:
        return self.feature_drifts

    def plot_feature_drift(self, feature, max_n_cat: int = 20):
        if self.X1 is None:
            raise ValueError('You must call the fit method before calling "get_feature_drift"')
        feature_index, feature_name = self._check_feature_param(feature, self.feature_names)
        if feature_index in self.cat_feature_indices:
            plot_drift_cat(self.X1.iloc[:,feature_index].values, self.X2.iloc[:,feature_index].values,
                           self.sample_weights1, self.sample_weights2, title=feature_name, max_n_cat=max_n_cat)
        else:
            plot_drift_num(self.X1.iloc[:,feature_index].values, self.X2.iloc[:,feature_index].values,
                           self.sample_weights1, self.sample_weights2, title=feature_name)

    @staticmethod
    def _check_feature_param(feature, feature_names):
        if isinstance(feature, str):
            if feature_names is None:
                raise ValueError(f'String parameter for "feature" but feature names not found in X1 (X2) columns')
            elif feature not in feature_names:
                raise ValueError(f'{feature} not found in X1 (X2) columns')
            else:
                return feature_names.index(feature), feature
        elif isinstance(feature, int):
            if feature_names is None:
                return feature, f'feature {feature}'
            else:
                return feature, feature_names[feature]
        else:
            raise ValueError(f'{feature} should be either of type str or int')

    def generate_html_report(self, output_path, max_n_cat: int = 20):
        # TODO : not ready
        DriftReportGenerator.generate(self, output_path, max_n_cat)

    def update(self, new_X):
        """usually new_X would update X2: the production X"""
        pass

    def plot_feature_contribs(self, n: int = 10, type: str = 'node_size'):
        # a voir si je veux rendre cette fonction plus générique
        if self.model_parser is None:
            raise ValueError('You need to run drift_explainer.fit before you can plot feature_contribs')
        # if n > n_features set n = n_features
        n = min(n, self.n_features)

        feature_contribs = self.model_parser.compute_feature_contribs(type=type)
        # sort by importance in terms of drift
        # sort in decreasing order according to sum of absolute values of feature_contribs
        order = np.abs(feature_contribs).sum(axis=1).argsort()[::-1].tolist()
        if self.feature_names is not None:
            ordered_names = [self.feature_names[i] for i in order]
        else:
            ordered_names = [f'feature {i}' for i in order]
        ordered_feature_contribs = feature_contribs[order, :]

        # there is a legend only in the case of multiclass classif (with type in ['mean', 'mean_norm'])
        if type in ['mean', 'mean_norm'] and self.model_parser.prediction_dim > 1:
            n_dim = self.model_parser.prediction_dim
            legend_labels = self.class_names
        else:
            n_dim = 1
            legend_labels = []

        # plot
        fig, ax = plt.subplots(figsize=(10, 10))

        X = np.arange(n)
        for i in range(n_dim):
            ax.barh(X + (n_dim-i-1)/(n_dim+1), ordered_feature_contribs[:n, i][::-1], height=1/(n_dim+1))
        ax.legend(legend_labels)
        ax.set_yticks(X + 1/(n_dim+1) * (n_dim-1)/2)
        ax.set_yticklabels(ordered_names[:n][::-1])
        ax.set_xlabel('Contribution to data drift', fontsize=15)
        plt.show()

    def plot_tree_drift(self, tree_idx: int, type: str = 'node_size'):
        if self.model_parser.node_weights1 is None:
            raise ValueError('You need to run drift_explainer.fit before calling plot_tree_drift')
        if type not in ['node_size', 'mean_norm', 'mean']:
            raise ValueError(f'Bad value for "type"')
        else:
            self.model_parser.trees[tree_idx].plot_drift(node_weights1=self.model_parser.node_weights1[tree_idx],
                                                         node_weights2=self.model_parser.node_weights2[tree_idx],
                                                         type=type,
                                                         feature_names=self.feature_names)

    def get_correction_weights(self, type='node_size'):
        if type == 'node_size':
            drift_corrector = TreeEnsembleDriftCorrector()
            drift_corrector.get_weights()
        elif type == 'adversarial':
            drift_corrector = AdversarialDriftCorrector()
        elif type == 'feature':
            drift_corrector = FeatureBasedDriftCorrector()
        else:
            raise ValueError(f'Bad value for "type" parameter: {type}')

    @staticmethod
    def _get_feature_names(X1, X2, model_parser: ITreeEnsembleParser):
        # we take feature names in X1 and X2 column names if provided
        if isinstance(X1, pd.DataFrame) and isinstance(X2, pd.DataFrame):
            if list(X1.columns) != list(X2.columns):
                raise ValueError('"X1.columns" and "X2.columns" are not equal')
            else:
                feature_names = list(X1.columns)

            # if catboost model, check that order of columns in X1 is consistent with feature names in catboost
            if model_parser.model_type in ['catboost.core.CatBoostClassifier', 'catboost.core.CatBoostRegressor']:
                if model_parser.feature_names is not None and model_parser.feature_names != feature_names:
                    raise ValueError('X1.columns and CatBoost "feature_names_" are ot equal')
        else:
            feature_names = None
        return feature_names

    @staticmethod
    def _get_cat_feature_indices(model_parser: ITreeEnsembleParser):
        if model_parser.model_type in ['catboost.core.CatBoostClassifier', 'catboost.core.CatBoostRegressor']:
            return model_parser.cat_feature_indices
        else:
            return []  # TODO: maybe add binary features to cat_feature_indices

    @staticmethod
    def _get_class_names(task, model_parser: ITreeEnsembleParser) -> List[str]:
        if task == 'regression':
            return []
        elif model_parser.model_type == 'catboost.core.CatBoostClassifier':
            return model_parser.class_names
        else:
            n_class = 2 if model_parser.prediction_dim == 1 else model_parser.prediction_dim
            return [str(i) for i in range(n_class)]

    def _parse_model(self, model, iteration_range, X):
        if safe_isinstance(model, 'catboost.core.CatBoostClassifier'):
            self.model_parser: ITreeEnsembleParser = CatBoostParser(model, 'catboost.core.CatBoostClassifier',
                                                                    iteration_range, X, task='classification')
        elif safe_isinstance(model, 'xgboost.core.Booster'):
            self.model_parser: ITreeEnsembleParser = XGBoostParser(model, 'xgboost.core.Booster', iteration_range, X)
        elif safe_isinstance(model, 'xgboost.sklearn.XGBClassifier'):
            # output of get_booster() is in binary format and universal among various XGBoost interfaces
            self.model_parser: ITreeEnsembleParser = XGBoostParser(model.get_booster(), 'xgboost.sklearn.XGBClassifier',
                                                                   iteration_range, X)
        elif safe_isinstance(model, "xgboost.sklearn.XGBRegressor"):
            self.model_parser: ITreeEnsembleParser = XGBoostParser(model.get_booster(), 'xgboost.sklearn.XGBRegressor',
                                                                   iteration_range, X)
        elif safe_isinstance(model, 'xgboost.sklearn.XGBRanker'):
            self.model_parser: ITreeEnsembleParser = XGBoostParser(model.get_booster(), 'xgboost.sklearn.XGBRanker',
                                                                   iteration_range, X)
        else:
            raise TypeError(f'The type of model {type(model).__name__} is not supported in DriftExplainer')
        if self.model_parser.task == 'ranking':
            DriftExplainer.logger.warning('A ranking model was passed to DriftExplainer. It will be treated similarly as'
                                          ' regression model but there is no warranty about the result')

    @staticmethod
    def _check_sample_weights(sample_weights, X):
        if sample_weights is None:
            return np.ones(X.shape[0])
        elif np.any(sample_weights < 0):
            raise ValueError("Elements in sample_weights must be non negative")
        elif np.sum(sample_weights) == 0:
            raise ValueError("The sum of sample_weights must be positive")
        else:
            return sample_weights

    @staticmethod
    def _check_X_shape(X1, X2):
        if X2.shape[1] != X1.shape[1]:
            raise ValueError('X1 and X2 do not have the same number of columns')
