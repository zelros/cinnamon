import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from typing import List, Tuple
from .abstract_drift_explainer import AbstractDriftExplainer
from ..common.math_utils import threshold_array


class AdversarialDriftExplainer(AbstractDriftExplainer):
    """
    Tool to study data drift between two datasets using a adversarial learning
    approach (i.e. training a classifier to discriminate between
    dataset 1 and dataset2). XGBClassifier is used as adversarial classifier.

    Parameters
    ----------
    n_splits : int (must be >= 2), optional (default=2)
        Number of folds in the cross validation.

    feature_subset : List[Union[int, str]], optional (default=None)
        Subset of features to consider in the training of the adversarial classifier.

    seed : int, optional (default=None)
        Random seed to set in order to get reproducible results.

    verbosity : bool, optional (default=True)
        Whether to print training logs of adversarial classifiers or not.

    max_depth : int, optional (default=6)
        "max_depth" parameter passed to XGBClassifier for each cross-validation
        model.

    learning_rate : float, optional (default=0.1)
        "learning_rate" parameter passed to XGBClassifier for each cross-validation
        model.

    tree_method : str, optional (default="auto")
        "tree_method" parameter passed to XGBClassifier for each cross-validation
        model.

    Attributes
    ----------
    cv_adversarial_models : List[XGBClassifier]
        List of cross-validated XGBClassifier models.

    kf_splits : List[Tuple[np.array, np.array]]
        List of the training-validation indexes (train_idx, val_idx) used in the
        cross-validation.

        Note: In order to learn the adversarial classifier, X1 and X2 are
        concatenated on axis=0, and indexes are reset after the concatenation.

    feature_drifts : list of dict
        Drift measures for each input feature in X.

    target_drift : dict
        Drift measures for the labels y.

    task : string
        Task corresponding to the (X, Y) data. Either "regression", "classification",
        or "ranking".

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
        y1 and y2 targets passed to the "fit" method

    sample_weights1, sample_weights2 : numpy arrays
        sample_weights1 and sample_weights2 arrays passed to the "fit" method
    """
    def __init__(self, n_splits: int = 2, feature_subset: List[str] = None, seed: int = None,
                 verbosity: bool = True, max_depth: int = 6, learning_rate: float = 0.1, tree_method: str = 'auto'):
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
        """
        Fit the adversarial drift explainer to dataset 1 and dataset 2.
        Only X1, X2, sample_weights1 and sample_weights2 are used to build the
        adversarial drift explainer. y1 and y2 are only only used if call to
        get_target_drift or plot_target_drift methods is made.

        Parameters
        ----------
        X1 : pandas dataframe of shape (n_samples, n_features)
            Dataset 1 inputs.

        X2 : pandas dataframe of shape (n_samples, n_features)
            Dataset 2 inputs.

        y1 : numpy array of shape (n_samples,), optional (default=None)
            Dataset 1 labels.
            If None, data drift is only analyzed based on inputs X1 and X2.

        y2 : numpy array of shape (n_samples,), optional (default=None)
            Dataset 2 labels
            If None, data drift is only analyzed based on inputs X1 and X2.

        sample_weights1: numpy array of shape (n_samples,), optional (default=None)
            Array of weights that are assigned to individual samples of dataset 1
            If None, then each sample of dataset 1 is given unit weight.

        sample_weights2: numpy array of shape (n_samples,), optional (default=None)
            Array of weights that are assigned to individual samples of dataset 2
            If None, then each sample of dataset 2 is given unit weight.

        Returns
        -------
        AdversarialDriftExplainer
            The fitted adversarial drift explainer.
        """
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
        """
        Compute drift values using the adversarial method. Here the drift values
        correspond to the means of the feature importance taken over the
        cross-validated adversarial classifiers.

        See the documentation in README for explanations about how it is computed,
        especially the slide presentation.

        Returns
        -------
        drift_values : numpy array
        """
        model_importances = [model.feature_importances_ for model in self.cv_adversarial_models]
        mean_importances = np.mean(model_importances, axis=0).reshape(-1, 1)
        return mean_importances

    def plot_adversarial_drift_values(self, n: int = 10):
        """
        Plot drift values computed using the adversarial method. Here the drift values
        correspond to the means of the feature importance taken over the n_splits
        cross-validated adversarial classifiers.

        See the documentation in README for explanations about how it is computed,
        especially the slide presentation.

        Parameters
        ----------
        n : interger, optional (default=10)
            Top n features to represent in the plot.

        Returns
        -------
        None
        """
        drift_values = self.get_adversarial_drift_values()
        self._plot_drift_values(drift_values, n, self.feature_subset)

    def get_adversarial_correction_weights(self, max_ratio: int = 10) -> np.array:
        """
        Compute weights for dataset 1 samples in order to correct data drift (more
        specifically in order to correct covariate shift).

        Given an adversarial classifier c: X -> [0, 1], the formula used to
        compute weights for a sample X_i is c(X_i) / (1 - c(X_i)) (cross-validation
        is used in order to compute weights for all dataset 1 samples).

        See the documentation in README for explanations about how it is computed,
        especially the slide presentation.

        Parameters
        ----------
        max_ratio: int, optional (default=10)
            Maximum ratio between two weights returned in correction_weights (weights
            are thresholded so that the ratio between two weights do not exceed
            max_ratio).

        Returns
        -------
        correction_weights : np.array
            Array of correction weights for the samples of dataset 1.
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
        # FIXME: problem here with cat features... and add it to the test
        return []

    @staticmethod
    def _get_class_names(task: str, y1, y2):
        # FIXME: problem here with feature names... and add it to the test
        return []

    @staticmethod
    def _get_task(y1: np.array, y2: np.array):
        if y1 is not None and y2 is not None:
            y = np.concatenate((y1, y2))
            if len(np.unique(y)) > 10:
                return 'regression'
            else:
                return 'classification'
