import pandas as pd
import numpy as np
from typing import List, Tuple

from .i_model_parser import IModelParser


class UnknownModelParser(IModelParser):

    def __init__(self, model, model_type):
        super().__init__()
        self.original_model = model
        self.model_type = model_type
        self.parse()

    def parse(self, iteration_range: Tuple[int, int] = None):
        pass

    def fit(self, X1: pd.DataFrame, X2: pd.DataFrame, sample_weights1: np.array, sample_weights2: np.array):
        pass

    def get_predictions(self, X: pd.DataFrame, prediction_type: str) -> np.array:
        return self.original_model.predict(X)

    def compute_prediction_based_drift_values(self) -> np.array:
        raise NotImplementedError('Will be implemented soon')

    def compute_tree_based_drift_values(self, type: str) -> np.array:
        self._not_tree_based_model_error()

    def check_tree_based_drift_values_sum(self, X1, X2, sample_weights1, sample_weights2):
        self._not_tree_based_model_error()

    def compute_tree_based_correction_weights(self, X1: pd.DataFrame, max_depth: int, max_ratio: int,
                                              sample_weights1: np.array) -> np.array:
        self._not_tree_based_model_error()

    def plot_tree_drift(self, tree_idx: int, type: str, feature_names: List[str]) -> None:
        self._not_tree_based_model_error()

    @staticmethod
    def _not_tree_based_model_error():
        raise NotImplementedError('Either passed model is not a tree based model or tree based drift values'
                                  'are not implemented yet for this model type')
