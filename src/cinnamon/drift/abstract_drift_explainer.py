import numpy as np
import pandas as pd
from typing import List, Union

from .drift_utils import compute_drift_cat, compute_drift_num, AbstractDriftMetrics
from ..common.logging import cinnamon_logger


class AbstractDriftExplainer:

    logger = cinnamon_logger.getChild('DriftExplainer')

    def __init__(self):
        self.feature_drifts = None  # drift of each input feature of the model
        self.target_drift = None  # drift of the target ground truth
        self.feature_names = None
        self.class_names = None
        self.cat_feature_indices = None  # indexes of categorical_features
        self.n_features = None
        self.task = None

        self.X1 = None
        self.X2 = None
        self.sample_weights1 = None
        self.sample_weights2 = None
        self.y1 = None
        self.y2 = None

    @staticmethod
    def _compute_feature_drifts(X1, X2, n_features, cat_feature_indices, sample_weights1, sample_weights2) -> List[AbstractDriftMetrics]:
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

    def get_target_drift(self) -> AbstractDriftMetrics:
        """
        Compute drift measures for the labels y.

        For classification :
        - Wasserstein distance
        - Result of Chi2 2 sample test

        For regression:
        - Difference of means
        - Wasserstein distance
        - Result of Kolmogorov 2 sample test

        See the documentation in README for explanations about how it is computed,
        especially the slide presentation.

        Returns
        -------
        target_drift : Union[DriftMetricsCat, DriftMetricsNum]
            Drift measures for the labels y.
        """
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

    def get_feature_drifts(self) -> List[AbstractDriftMetrics]:
        """
        Compute drift measures for all features in X.

        For numerical features:
        - Difference of means
        - Wasserstein distance
        - Result of Kolmogorov 2 sample test

        For categorial features (not supported currently. No categorical
        feature allowed):
        - Wasserstein distance
        - Result of Chi2 two sample test

        See the documentation in README for explanations about how it is computed,
        especially the slide presentation.

        Returns
        -------
        feature_drift: list of Union[DriftMetricsCat, DriftMetricsNum].
            Drift measures for each input feature in X.
        """
        if self.feature_drifts is None:
            self.feature_drifts = self._compute_feature_drifts(self.X1, self.X2, self.n_features, self.cat_feature_indices,
                                                               self.sample_weights1, self.sample_weights2)
        return self.feature_drifts

    def get_feature_drift(self, feature: Union[int, str]) -> AbstractDriftMetrics:
        """
        Compute drift measures for a given feature in X.

        For numerical feature:
        - Difference of means
        - Wasserstein distance
        - Result of Kolmogorov 2 sample test

        For categorial feature (not supported currently. No categorical
        feature allowed):
        - Wasserstein distance
        - Result of Chi2 two sample test

        See the documentation in README for explanations about how it is computed,
        especially the slide presentation.

        Parameters
        ----------
        feature : Union[int, str]
            Either the column index of the name of the feature.

        Returns
        -------
        feature_drift: Union[DriftMetricsCat, DriftMetricsNum]
            Drift measures of the input feature.
        """
        if self.X1 is None:
            raise ValueError('You must call the fit method before calling "get_feature_drift"')
        feature_index, feature_name = self._check_feature_param(feature, self.feature_names)
        return self.get_feature_drifts()[feature_index]

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

    """
    def generate_html_report(self, output_path, max_n_cat: int = 20):
        # FIXME : not ready
        pass

    def update(self, new_X):
        '''usually new_X would update X2: the production X'''
        pass
    """

    def _check_fit_arguments(self, X1, X2, y1, y2, sample_weights1, sample_weights2, cat_feature_indices):
        self.sample_weights1 = self._check_sample_weights(sample_weights1, X1)
        self.sample_weights2 = self._check_sample_weights(sample_weights2, X2)
        self.X1, self.X2 = self._check_X(X1, X2)
        self._check_X_shape(self.X1, self.X2)
        self.y1 = y1
        self.y2 = y2
        self.cat_feature_indices = cat_feature_indices

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
