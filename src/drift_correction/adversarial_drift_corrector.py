from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd

from .i_drift_corrector import IDriftCorrector
from ..common.math_utils import threshold_array


class AdversarialDriftCorrector(IDriftCorrector):

    def __init__(self, X1, X2, n_splits, feature_subset, max_depth, max_ratio, seed):
        self.X1 = X1
        self.X2 = X2
        self.n_splits = n_splits
        self.feature_subset = feature_subset if feature_subset is not None else X1.columns.to_list()
        self.max_depth = max_depth
        self.max_ratio = max_ratio
        self.seed = seed
        self.cv_models = None

    def get_weights(self, return_object: bool):
        weights, self.cv_models = self._get_weights(self.X1, self.X2, self.n_splits, self.feature_subset,
                                                    self.max_depth, self.max_ratio, self.seed)
        if return_object:
            return weights, self
        else:
            return weights

    @staticmethod
    def _get_weights(X1: pd.DataFrame, X2: pd.DataFrame, n_splits: int, feature_subset, max_depth: int, max_ratio: int,
                     seed: int):
        # TODO: handle feature subset if int provided
        sample_weights = np.array([X2.shape[0] / X1.shape[0]] * X1.shape[0] + [1.0] * X2.shape[0])
        y = np.array([0] * X1.shape[0] + [1] * X2.shape[0])
        X = pd.concat([X1[feature_subset], X2[feature_subset]], axis=0, ignore_index=True)
        kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
        weights = np.zeros(X1.shape[0])
        cv_models = []
        for train_idx, val_idx in kf.split(X):
            X_train, X_valid = X.loc[train_idx], X.loc[val_idx]
            y_train, y_valid = y[train_idx], y[val_idx]
            sample_weights_train, sample_weights_valid = sample_weights[train_idx], sample_weights[val_idx]
            clf = XGBClassifier(n_estimators=10000,
                                booster="gbtree",
                                objective="binary:logistic",
                                learning_rate=0.1,
                                max_depth=max_depth,
                                use_label_encoder=False,
                                seed=seed)
            clf.fit(X=X_train, y=y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=20,
                    sample_weight=sample_weights_train, verbose=10, eval_metric=['error', 'auc', 'logloss'],
                    sample_weight_eval_set=[sample_weights_valid])
            cv_models.append(clf)
            val_idx_in_X1 = val_idx[val_idx < X1.shape[0]]
            pred_proba = clf.predict_proba(X_valid.loc[val_idx_in_X1])[:, 1]
            weights[val_idx_in_X1] = threshold_array(pred_proba / (1 - pred_proba), max_ratio)
        return weights, cv_models
