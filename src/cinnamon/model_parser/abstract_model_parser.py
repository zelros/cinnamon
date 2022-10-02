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
        else:
            raise ValueError(f'Bad value for prediction_type: {prediction_type}')
    
    def predict_raw(self, X: pd.DataFrame) -> np.array:
        return self.original_model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        return self.original_model.predict_proba(X)

    def get_prediction_dim(self, X1=None) -> int:
        if self.prediction_dim:
            return self.prediction_dim
        else:
            temp_dim = np.array(self.get_predictions(X1, prediction_type='raw')).squeeze().shape
            if len(temp_dim) == 1:
                return 1
            elif len(temp_dim) == 2 and temp_dim[1] <= 2:
                return 1
            elif len(temp_dim) == 2:
                temp_dim[1]
            else:
                raise ValueError(f'Can not infer the predicted dimension of the output from the model provided')
