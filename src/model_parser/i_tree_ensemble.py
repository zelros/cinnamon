import pandas as pd
import numpy as np


class ITreeEnsembleParser:
    def __init__(self):
        self.model_type = None
        self.original_model = None
        self.n_trees = None
        self.max_depth = None
        self.cat_feature_indices = None
        self.n_features = None
        self.trees = None
        self.prediction_dim = None
        self.model_objective = None
        self.task = None

        # specifiv to catboost
        self.class_names = None
        self.feature_names = None

    def parse(self, model):
        pass

    def get_predictions(self, X: pd.DataFrame, prediction_type: str):
        pass

    def get_node_weights(self, X: pd.DataFrame, sample_weights: np.array):
        pass
