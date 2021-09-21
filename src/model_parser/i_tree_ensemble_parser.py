import pandas as pd
import numpy as np


class ITreeEnsembleParser:
    def __init__(self):
        self.original_model = None
        self.model_type = None
        self.iteration_range = None
        self.n_trees = None
        self.max_depth = None
        self.cat_feature_indices = None
        self.n_features = None
        self.trees = None
        self.prediction_dim = None
        self.model_objective = None
        self.task = None
        self.n_iterations = None
        self.base_score = None  # bias
        self.node_weights1 = None
        self.node_weights2 = None

        # specific to catboost
        self.class_names = None
        self.feature_names = None

    def parse(self, iteration_range, X):
        pass

    def predict_leaf(self, X: pd.DataFrame) -> np.array:
        pass

    def predict_raw(self, X: pd.DataFrame) -> np.array:
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        pass

    def predict_leaf_with_model_parser(self, X: pd.DataFrame) -> np.array:
        pass
