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


def reverse_binary_representation(n: int, n_bits: int):
    # return the integer obtained via reversing the order of the binary representation of the integer
    # -------- example -----------
    # n = 4, n_bits = 3
    # binary representation = '100', reversed in '001'
    # hence the returned integer is 1
    # ---------------------------
    assert 0 <= n < 2 ** n_bits
    binary_representation = np.binary_repr(n, width=n_bits)
    # we do the loop from left to right so no need to use binary_representation[::-1]
    return sum([2**i for i, b in enumerate(binary_representation) if b == '1'])
