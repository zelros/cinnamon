import numpy as np


def threshold(x: float, max_ratio: float):
    return min(max_ratio, max(x, 1/max_ratio))


def threshold_array(a: np.array, max_ratio: float):
    a[a > max_ratio] = max_ratio
    a[a < 1/max_ratio] = 1/max_ratio
    return a


def softmax(a: np.array):
    """
    :param a: array of log_softmax (should have ndim = 2)
    :return:
    """
    return np.exp(a) / np.sum(np.exp(a), axis=1)[:, None]


def sigmoid(a: np.array):
    '''
    :param a: array of logit (should have ndim = 1)
    :return:
    '''
    return np.exp(a) / (1 + np.exp(a))
