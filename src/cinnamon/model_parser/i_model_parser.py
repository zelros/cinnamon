import pandas as pd
import numpy as np
from typing import List, Tuple


class IModelParser:
    def __init__(self):
        self.original_model = None
        self.model_type = None
        self.cat_feature_indices = None
        self.n_features = None
        self.prediction_dim = None
        self.task = None
        self.class_names = None
        self.feature_names = None
        self.iteration_range = None

    def parse(self, iteration_range: Tuple[int, int]):
        pass

    def fit(self, X1: pd.DataFrame, X2: pd.DataFrame, sample_weights1: np.array, sample_weights2: np.array):
        pass

    def get_predictions(self, X: pd.DataFrame, prediction_type: str) -> np.array:
        pass

    def compute_prediction_based_drift_values(self) -> np.array:
        pass

    def compute_tree_based_drift_values(self, type: str) -> np.array:
        pass

    def compute_tree_based_correction_weights(self, X1: pd.DataFrame, max_depth: int, max_ratio: int,
                                              sample_weights1: np.array) -> np.array:
        pass

    def plot_tree_drift(self, tree_idx: int, type: str, feature_names: List[str]) -> None:
        pass

    def check_tree_based_drift_values_sum(self, X1, X2, sample_weights1, sample_weights2) -> None:
        pass
