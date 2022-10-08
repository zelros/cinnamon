import pandas as pd
import numpy as np
from typing import List, Tuple

from .abstract_model_parser import AbstractModelParser
from ..common.dev_utils import has_method
from ..common.math_utils import log_softmax


class ModelAgnosticModelParser(AbstractModelParser):

    def __init__(self, model, model_type, task: str):
        super().__init__(model, model_type)
        self.task = task

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        return self.original_model.predict_proba(X)

    def predict_raw(self, X: pd.DataFrame):
        if self.task in ['regression', 'ranking']:
            return self.original_model.predict(X)
        else:
            if has_method(self.original_model, 'predict_raw'):
                self.original_model.predict_raw(X)
            else:
                return log_softmax(self.original_model.predict_proba(X))

    def predict_class(self, X: pd.DataFrame):
        # drift on predicted class is not handled in current version
        if self.task == 'classification':
            return self.original_model.predict(X)
