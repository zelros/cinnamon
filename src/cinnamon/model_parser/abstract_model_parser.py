import pandas as pd
import numpy as np
from typing import List, Tuple


class AbstractModelParser:
    def __init__(self, model, model_type):
        self.original_model = model
        self.model_type = model_type
        self.cat_feature_indices = None
        self.n_features = None
        self.prediction_dim = None
        self.task = None
        self.class_names = None
        self.feature_names = None
        self.iteration_range = None

    def parse(self):
        pass

    def fit(self, X1: pd.DataFrame, X2: pd.DataFrame, sample_weights1: np.array, sample_weights2: np.array):
        pass

    def get_predictions(self, X: pd.DataFrame, prediction_type: str) -> np.array:
        # return array of shape (nb. obs, nb. class) for multiclass and shape array of shape (nb. obs, )
        # for binary class and regression
        """

        :param X:
        :param prediction_type: "raw" or "proba"
        :return:
        """
        # array of shape (nb. obs, nb. class) for multiclass and shape array of shape (nb. obs, )
        # for binary class and regression
        if prediction_type == 'raw':
            return self.predict_raw(X)
        elif prediction_type == 'proba':
            return self.predict_proba(X)
        elif prediction_type == 'class':
            return self.predict_class(X)
        else:
            raise ValueError(f'Bad value for prediction_type: {prediction_type}')
    
    def predict_raw(self, X: pd.DataFrame) -> np.array:
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        pass

    def predict_class(self, X: pd.DataFrame) -> np.array:
        pass

    def get_prediction_dim(self, X1=None) -> int:
        if not self.prediction_dim:
            if self.task in ['regression', 'ranking']:
                self.prediction_dim = 1
            else:        
                pred_shape = self.predict_proba(X1).shape
                if len(pred_shape) == 1:
                    self.prediction_dim = 1
                elif len(pred_shape) == 2 and pred_shape[1] <= 2:
                    self.prediction_dim = 1
                elif len(pred_shape) == 2:
                    self.prediction_dim = pred_shape[1]
                else:
                    raise ValueError(f'Can not infer the predicted dimension of the output from the model provided')
        return self.prediction_dim
