import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, mean_squared_error, explained_variance_score, roc_auc_score, accuracy_score
from dataclasses import dataclass


def compute_classification_metrics(y_true: np.array, y_pred: np.array, sample_weights: np.array) -> dict:
    # TODO: add more metrics here: accuracy, AUC, etc.
    return {'log_loss': log_loss(y_true, y_pred, sample_weight=sample_weights)}


def compute_regression_metrics(y_true: np.array, y_pred: np.array, sample_weights: np.array) -> dict:
    # TODO: add more metrics here
    return {'mse': mean_squared_error(y_true, y_pred, sample_weight=sample_weights),
            'explained_variance': explained_variance_score(y_true, y_pred, sample_weight=sample_weights)}


@dataclass(frozen=True)
class StatisticalTestResultBase:
    statistic: float
    pvalue: float


@dataclass(frozen=True)
class Chi2TestResult(StatisticalTestResultBase):
    dof: int = None
    contingency_table: pd.DataFrame = None

    def __eq__(self, other):
        if not isinstance(other, Chi2TestResult):
            return False
        elif self.statistic != other.statistic or self.pvalue != other.pvalue or self.dof != other.dof:
            return False
        elif self.contingency_table.equals(other.contingency_table):
            return False
        else:
            return True
