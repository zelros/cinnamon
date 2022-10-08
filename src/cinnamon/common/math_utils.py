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


def divide_with_0(a: np.array, b: np.array) -> np.array:
    c = np.zeros_like(b)
    c[b>0] = a[b>0] / b[b>0]
    return c


def map_array_to_bins(a: np.array, bin_edges: np.array) -> np.array:
    '''
    example:
    a = np.array([5, 3, 3, 0])
    bin_edges = np.array([0, 2, 4, 6])
    -> result = np.array([3, 1, 1, 0])
    
    remark: bin_edes should be sorted in increasing order
    '''
    bins_ids = np.digitize(a, bin_edges)
    bins_ids[bins_ids == len(bin_edges)] = len(bin_edges) - 1
    return bins_ids - 1


def log_softmax(a: np.array) -> np.array:
    # from an array of predicted probabilities in rows, compute the rowwise log softmax of predictions
    # so that the mean of each row is equal to 0.
    log_predictions = np.log(a)
    return log_predictions - np.mean(log_predictions, axis=1)[:, None]
