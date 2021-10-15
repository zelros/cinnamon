import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

from .drift_utils import compute_drift_cat, compute_drift_num, plot_drift_cat, plot_drift_num
from ..report.drift_report_generator import DriftReportGenerator

logging.basicConfig(format='%(levelname)s:%(asctime)s - (%(pathname)s) %(message)s', level=logging.INFO)


class DriftExplainerABC:

    logger = logging.getLogger('DriftExplainer')

    def __init__(self):
        self.feature_drifts = None  # drift of each input feature of the model
        self.target_drift = None  # drift of the target ground truth
        self.feature_names = None
        self.class_names = None
        self.cat_feature_indices = None  # indexes of categorical_features
        self.n_features = None
        self.task = None

        # usefull for the report when I export the dataset (may not be usefull if I stock only the necessary info
        # to make the plot
        self.X1 = None  # same
        self.X2 = None  # same
        self.sample_weights1 = None  # same
        self.sample_weights2 = None  # same
        self.y1 = None  # same
        self.y2 = None  # same

    @staticmethod
    def _compute_feature_drifts(X1, X2, n_features, cat_feature_indices, sample_weights1, sample_weights2) -> List[dict]:
        #DriftExplainer.logger.info('Evaluate univariate drift of each feature')
        feature_drifts = []
        for i in range(n_features):
            if i in cat_feature_indices:
                feature_drift = compute_drift_cat(X1.iloc[:, i].values, X2.iloc[:, i].values,
                                                  sample_weights1, sample_weights2)
            else:
                feature_drift = compute_drift_num(X1.iloc[:,i].values, X2.iloc[:,i].values,
                                                  sample_weights1, sample_weights2)
            feature_drifts.append(feature_drift)
        return feature_drifts

    def get_target_drift(self):
        if self.target_drift is not None:
            return self.target_drift

        if self.y1 is None or self.y2 is None:
            self._raise_no_target_error()
        else:
            if self.task == 'classification':
                self.target_drift = compute_drift_cat(self.y1, self.y2, self.sample_weights1, self.sample_weights2)
            elif self.task in ['regression', 'ranking']:
                self.target_drift = compute_drift_num(self.y1, self.y2, self.sample_weights1, self.sample_weights2)
        return self.target_drift

    @staticmethod
    def _raise_no_target_error():
        raise ValueError('Either y1 or y2 was not passed in DriftExplainer.fit')

    def plot_target_drift(self, max_n_cat: int = 20, figsize=(7, 5)):
        if self.y1 is None or self.y2 is None:
            raise ValueError('"y1" or "y2" argument was not passed to drift_explainer.fit method')
        if self.task == 'classification':
            plot_drift_cat(self.y1, self.y2, self.sample_weights1, self.sample_weights2, title='target',
                           max_n_cat=max_n_cat, figsize=figsize)
        elif self.task == 'regression':
            plot_drift_num(self.y1, self.y2, self.sample_weights1, self.sample_weights2, title='target',
                           figsize=figsize)

    def get_feature_drifts(self) -> List[dict]:
        if self.feature_drifts is None:
            self.feature_drifts = self._compute_feature_drifts(self.X1, self.X2, self.n_features, self.cat_feature_indices,
                                                               self.sample_weights1, self.sample_weights2)
        return self.feature_drifts

    def get_feature_drift(self, feature) -> dict:
        if self.X1 is None:
            raise ValueError('You must call the fit method before calling "get_feature_drift"')
        feature_index, feature_name = self._check_feature_param(feature, self.feature_names)
        return self.get_feature_drifts()[feature_index]

    def plot_feature_drift(self, feature, max_n_cat: int = 20, figsize=(7, 5), as_discrete=False):
        if self.X1 is None:
            raise ValueError('You must call the fit method before calling "get_feature_drift"')
        feature_index, feature_name = self._check_feature_param(feature, self.feature_names)
        if feature_index in self.cat_feature_indices or as_discrete:
            plot_drift_cat(self.X1.iloc[:,feature_index].values, self.X2.iloc[:,feature_index].values,
                           self.sample_weights1, self.sample_weights2, title=feature_name, max_n_cat=max_n_cat,
                           figsize=figsize)
        else:
            plot_drift_num(self.X1.iloc[:,feature_index].values, self.X2.iloc[:,feature_index].values,
                           self.sample_weights1, self.sample_weights2, title=feature_name, figsize=figsize)

    def _plot_drift_values(self, drift_values: np.array, n: int, feature_names: List[str]):
        # threshold n if n > drift_values.shape[0]
        n = min(n, drift_values.shape[0])

        # sort by importance in terms of drift
        # sort in decreasing order according to sum of absolute values of drift_values
        order = np.abs(drift_values).sum(axis=1).argsort()[::-1].tolist()
        ordered_names = [feature_names[i] for i in order]
        ordered_drift_values = drift_values[order, :]

        n_dim = drift_values.shape[1]
        legend_labels = self.class_names if n_dim > 1 else []

        # plot
        fig, ax = plt.subplots(figsize=(10, 10))
        X = np.arange(n)
        for i in range(n_dim):
            ax.barh(X + (n_dim-i-1)/(n_dim+1), ordered_drift_values[:n, i][::-1], height=1/(n_dim+1))
        ax.legend(legend_labels)
        ax.set_yticks(X + 1/(n_dim+1) * (n_dim-1)/2)
        ax.set_yticklabels(ordered_names[:n][::-1])
        ax.set_xlabel('Drift values per feature', fontsize=15)
        plt.show()

    @staticmethod
    def _check_feature_param(feature, feature_names):
        if isinstance(feature, str):
            if feature not in feature_names:
                raise ValueError(f'{feature} not found in X1 (X2) columns')
            else:
                return feature_names.index(feature), feature
        elif isinstance(feature, int):
            return feature, feature_names[feature]
        else:
            raise ValueError(f'{feature} should be either of type str or int')

    def generate_html_report(self, output_path, max_n_cat: int = 20):
        # TODO : not ready
        DriftReportGenerator.generate(self, output_path, max_n_cat)

    def update(self, new_X):
        """usually new_X would update X2: the production X"""
        pass

    def _check_fit_arguments(self, X1, X2, y1, y2, sample_weights1, sample_weights2):
        self.sample_weights1 = self._check_sample_weights(sample_weights1, X1)
        self.sample_weights2 = self._check_sample_weights(sample_weights2, X2)
        self.X1, self.X2 = self._check_X(X1, X2)
        self._check_X_shape(self.X1, self.X2)
        self.y1 = y1
        self.y2 = y2

    @staticmethod
    def _check_sample_weights(sample_weights, X):
        if sample_weights is None:
            return np.ones(X.shape[0])
        elif np.any(sample_weights < 0):
            raise ValueError("Elements in sample_weights must be non negative")
        elif np.sum(sample_weights) == 0:
            raise ValueError("The sum of sample_weights must be positive")
        else:
            return sample_weights * len(sample_weights) / np.sum(sample_weights)

    @staticmethod
    def _check_X(X1, X2):
        return pd.DataFrame(X1), pd.DataFrame(X2)

    @staticmethod
    def _check_X_shape(X1, X2):
        if X2.shape[1] != X1.shape[1]:
            raise ValueError('X1 and X2 do not have the same number of columns')
