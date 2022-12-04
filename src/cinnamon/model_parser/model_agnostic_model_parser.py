import pandas as pd
import numpy as np
from typing import List, Tuple

from sklearn.model_selection import PredefinedSplit

from .abstract_model_parser import AbstractModelParser
from ..common.dev_utils import has_method
from ..common.math_utils import log_softmax, logit


class ModelAgnosticModelParser(AbstractModelParser):

    def __init__(self, model, model_type, task: str):
        super().__init__(model, model_type)
        self.task = task

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        predicted_proba = self.original_model.predict_proba(X).squeeze()
        if predicted_proba.ndim == 1:
            return predicted_proba
        else:
            assert predicted_proba.ndim == 2
            if predicted_proba.shape[1] == 2:
                return predicted_proba[:, 1]
            else:
                return predicted_proba

    def predict_raw(self, X: pd.DataFrame):
        if self.task in ['regression', 'ranking']:
            return self.original_model.predict(X)
        else:
            if has_method(self.original_model, 'predict_raw'):
                self.original_model.predict_raw(X)
            else:
                if self.get_prediction_dim(X) == 1:
                    return logit(self.predict_proba(X))
                else:
                    return log_softmax(self.predict_proba(X))

    def predict_class(self, X: pd.DataFrame):
        if self.task == 'classification':
            return self.original_model.predict(X)
        else:
            raise AttributeError('predicted class only supported for "classification" algorithms.')
