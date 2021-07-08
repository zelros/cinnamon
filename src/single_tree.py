import numpy as np


class BinaryTree:
    def __init__(self, children_left: np.array, children_right: np.array, children_default: np.array,
                 split_features_index: np.array, split_values: np.array,
                 values: np.array, train_sample_weights: np.array):
        self.children_left = children_left
        self.children_right = children_right
        self.children_default = children_default
        self.split_features_index = split_features_index
        self.split_values = split_values
        self.values = values
        self.train_sample_weights = train_sample_weights  # does it really correspond to train samples ? (or train + valid ?)
        self.n_leaves = len(train_sample_weights) - len(self.children_left)

    def compute_feature_drift_values(self, sample_weights1: np.array, sample_weights2: np.array,
                                     n_features: int) -> dict:
        """

        :param sample_weights1:
        :param sample_weights2: Usually correspond to train dataset
        :param n_features:
        :return:
        """
        assert len(sample_weights1) == len(sample_weights2)
        split_drift_values = self._compute_split_drift_values(sample_weights1, sample_weights2)
        feature_drift_values = np.zeros((n_features, split_drift_values.shape[1]))
        for feature_index, drift_values in zip(self.split_features_index, split_drift_values):
            feature_drift_values[feature_index, :] += drift_values
        return feature_drift_values

    def _compute_split_drift_values(self, sample_weights1: np.array, sample_weights2: np.array):
        split_contribs1 = self._compute_split_contribs(sample_weights1)
        split_contribs2 = self._compute_split_contribs(sample_weights2)
        split_drift_values = split_contribs2 - split_contribs1
        return split_drift_values

    def _compute_split_contribs(self, sample_weights: np.array) -> np.array:
        """
        :param sample_weights:
        :return:
        """
        n_nodes = len(self.children_left)
        sample_weight_fractions = sample_weights / sample_weights[0]
        split_contribs = np.zeros((n_nodes, self.values.shape[1]))
        for i in range(n_nodes):
            split_contribs[i] = (sample_weight_fractions[self.children_left[i]] * self.values[self.children_left[i]] +
                                sample_weight_fractions[self.children_right[i]] * self.values[self.children_right[i]] -
                                sample_weight_fractions[i] * self.values[i])
        return split_contribs
