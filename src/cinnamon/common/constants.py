from enum import Enum

NUMPY_atol = 1e-8
DEFAULT_atol = 1e-12
DEFAULT_rtol = 1e-8


class TreeBasedDriftValueType(Enum):
    NODE_SIZE = "node_size"
    MEAN = "mean"
    MEAN_NORM = "mean_norm"


class ModelAgnosticDriftValueType(Enum):
    MEAN = 'mean'
    WASSERSTEIN = 'wasserstein'
