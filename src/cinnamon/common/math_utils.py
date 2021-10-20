import numpy as np
from sklearn.metrics import log_loss, mean_squared_error, explained_variance_score, roc_auc_score, accuracy_score


def threshold(x: float, max_ratio: float):
    return min(max_ratio, max(x, 1/max_ratio))


def threshold_array(a: np.array, max_ratio: float):
    a[a > max_ratio] = max_ratio
    a[a < 1/max_ratio] = 1/max_ratio
    return a


def softmax(a: np.array):
    """
    :param a: array of log_softmax
    :return:
    """
    return np.exp(a) / np.sum(np.exp(a), axis=1)[:, None]


def compute_classification_metrics(y_true: np.array, y_pred: np.array, sample_weights: np.array) -> dict:
    # TODO: add more metrics here: accuracy, AUC, etc.
    return {'log_loss': log_loss(y_true, y_pred, sample_weight=sample_weights)}


def compute_regression_metrics(y_true: np.array, y_pred: np.array, sample_weights: np.array) -> dict:
    # TODO: add more metrics here
    return {'mse': mean_squared_error(y_true, y_pred, sample_weight=sample_weights),
            'explained_variance': explained_variance_score(y_true, y_pred, sample_weight=sample_weights)}
