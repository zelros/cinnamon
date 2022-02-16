from enum import Enum

NUMPY_atol = 1e-8
FLOAT_atol = 1e-12


class TreeBasedDriftValueType(Enum):
    NODE_SIZE = "node_size"
    MEAN = "mean"
    MEAN_NORM = "mean_norm"
