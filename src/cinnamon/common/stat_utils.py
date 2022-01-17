import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, mean_squared_error, explained_variance_score, roc_auc_score, accuracy_score
from dataclasses import dataclass
from pandas.testing import assert_frame_equal


def compute_classification_metrics(y_true: np.array, y_pred: np.array, sample_weights: np.array) -> dict:
    # TODO: add more metrics here: accuracy, AUC, etc.
    return {'log_loss': log_loss(y_true, y_pred, sample_weight=sample_weights)}


def compute_regression_metrics(y_true: np.array, y_pred: np.array, sample_weights: np.array) -> dict:
    # TODO: add more metrics here
    return {'mse': mean_squared_error(y_true, y_pred, sample_weight=sample_weights),
            'explained_variance': explained_variance_score(y_true, y_pred, sample_weight=sample_weights)}


@dataclass(frozen=True)
class BaseStatisticalTestResult:
    statistic: float
    pvalue: float

    def assert_equal(self, other):
        assert isinstance(other, BaseStatisticalTestResult)
        assert self.statistic == other.statistic
        assert self.pvalue == other.pvalue

@dataclass(frozen=True)
class Chi2TestResult(BaseStatisticalTestResult):
    dof: int = None
    contingency_table: pd.DataFrame = None

## see later if this functin is needed
#    def __eq__(self, other):
#        if not isinstance(other, Chi2TestResult):
#            return False
#        elif self.statistic != other.statistic or self.pvalue != other.pvalue or self.dof != other.dof:
#            return False
#        elif not self.contingency_table.equals(other.contingency_table):
#            return False
#        else:
#            return True

    def assert_equal(self, other):
        assert isinstance(other, Chi2TestResult)
        assert self.statistic == other.statistic
        assert self.pvalue == other.pvalue
        assert self.dof == other.dof
        assert_frame_equal(self.contingency_table, other.contingency_table)
