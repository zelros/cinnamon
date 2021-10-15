import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from typing import List, Tuple
from .drift_explainer_abc import DriftExplainerABC
from ..common.math_utils import threshold_array


class AdversarialDriftExplainer(DriftExplainerABC):

    def __init__(self, n_splits: int = 2, feature_subset: List[str] = None, max_depth: int = 6, seed: int = None,
                 verbosity: bool = True, learning_rate: float = 0.1, tree_method: str = 'auto'):
        super().__init__()
        self.n_splits = n_splits
        self.feature_subset = feature_subset
        self.max_depth = max_depth
        self.seed = seed
        self.verbosity = verbosity
        self.learning_rate = learning_rate
        self.tree_method = tree_method
        self.cv_adversarial_models: List[XGBClassifier] = []
        self.kf_splits: List[Tuple[np.array, np.array]] = []

    def fit(self, X1: pd.DataFrame, X2: pd.DataFrame, y1: np.array = None, y2: np.array = None,
            sample_weights1: np.array = None, sample_weights2: np.array = None):

        # Check arguments and save them as attributes
        self._check_fit_arguments(X1, X2, y1, y2, sample_weights1, sample_weights2)

        # set some class attributes
        self.n_features = self._get_n_features(self.X1)
        self.feature_names = self._get_feature_names(self.X1)
        self.cat_feature_indices = self._get_cat_feature_indices()
        self.class_names = self._get_class_names(self.task, self.y1, self.y2)
        self.feature_subset = self._get_feature_subset(self.feature_subset, self.X1)
        self.task = self._get_task(y1, y2)

        # fit adversarial classifier on data
        self.cv_adversarial_models, self.kf_splits = \
            self._build_adversarial_models(self.X1, self.X2, self.sample_weights1, self.sample_weights2,
                                           self.n_splits, self.feature_subset, self.max_depth, self.seed,
                                           self.verbosity, self.learning_rate, self.tree_method)
        return self

    def get_adversarial_drift_values(self):
        model_importances = [model.feature_importances_ for model in self.cv_adversarial_models]
        mean_importances = np.mean(model_importances, axis=0).reshape(-1, 1)
        return mean_importances

    def plot_adversarial_drift_values(self, n: int = 10):
        drift_values = self.get_adversarial_drift_values()
        self._plot_drift_values(drift_values, n, self.feature_subset)

    def get_adversarial_correction_weights(self, max_ratio: int = 10) -> np.array:
        """
        rmk:
        - for weights we always use the normalization: weights = weights * len(weights) / np.sum(weights)
        (because weights = weights / np.sum(weights) lead to small values that may create bugs)
        - if sample weights are passed to fit, the returned weights are the result of
        sample_weights * correction_weights (unique solution de tte façon car sinon il faudrait renvoyer des poids
        centrés en 1... galère pour l'utilisateur)
        :param max_ratio:
        :return:
        """
        correction_weights = np.zeros(self.X1.shape[0])
        for i, (train_idx, val_idx) in enumerate(self.kf_splits):
            val_idx_in_X1 = val_idx[val_idx < self.X1.shape[0]]
            pred_proba = (self.cv_adversarial_models[i]
                              .predict_proba(self.X1.iloc[val_idx_in_X1][self.feature_subset])[:, 1])
            correction_weights[val_idx_in_X1] = threshold_array(pred_proba / (1 - pred_proba), max_ratio)
        new_weights = self.sample_weights1 * correction_weights  # update initial sample_weights with the correction
        return new_weights * len(new_weights) / np.sum(new_weights)

    def _get_feature_subset(self, feature_subset: List[str], X1: pd.DataFrame) -> List[str]:
        if feature_subset is not None:
            feature_subset = [self._check_feature_param(feature, self.feature_names)[1] for feature in feature_subset]
        else:
            feature_subset = X1.columns.to_list()
        return feature_subset

    @staticmethod
    def _build_adversarial_models(X1: pd.DataFrame, X2: pd.DataFrame, sample_weights1: np.array,
                                  sample_weights2: np.array, n_splits: int, feature_subset: List[str],
                                  max_depth: int, seed: int, verbosity: bool, learning_rate: float, tree_method: str):
        sample_weights = np.concatenate((sample_weights1 * len(sample_weights2) / len(sample_weights1),
                                         sample_weights2))
        y = np.array([0] * X1.shape[0] + [1] * X2.shape[0])
        X = pd.concat([X1[feature_subset], X2[feature_subset]], axis=0, ignore_index=True)
        kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
        cv_adversarial_models = []
        kf_splits = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_valid = y[train_idx], y[val_idx]
            sample_weights_train, sample_weights_valid = sample_weights[train_idx], sample_weights[val_idx]
            clf = XGBClassifier(n_estimators=10000,
                                booster="gbtree",
                                objective="binary:logistic",
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                tree_method=tree_method,
                                use_label_encoder=False,
                                seed=seed)
            log_frequency = 10 if verbosity else 0
            clf.fit(X=X_train, y=y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=20,
                    sample_weight=sample_weights_train, verbose=log_frequency, eval_metric=['error', 'auc', 'logloss'],
                    sample_weight_eval_set=[sample_weights_valid])
            if verbosity:
                # TODO: print some logs about the trained adversarial model
                pass
            cv_adversarial_models.append(clf)
            kf_splits.append((train_idx, val_idx))
        return cv_adversarial_models, kf_splits  # TODO: add performance metrics of adversarial models ?

    @staticmethod
    def _get_n_features(X1: pd.DataFrame):
        return X1.shape[1]

    @staticmethod
    def _get_feature_names(X1: pd.DataFrame):
        return X1.columns.to_list()

    @staticmethod
    def _get_cat_feature_indices():
        # TODO: problem here with cat features...
        return []

    @staticmethod
    def _get_class_names(task: str, y1, y2):
        # TODO: problem here with feature names...
        return []

    @staticmethod
    def _get_task(y1: np.array, y2: np.array):
        if y1 is None or y2 is None:
            raise ValueError('Either y1 or y2 was not passed in AdversarialDriftExplainer.fit')
        y = np.concatenate((y1, y2))
        if len(np.unique(y)) > 10:
            return 'regression'
        else:
            return 'classification'
