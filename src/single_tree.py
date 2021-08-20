import numpy as np


class BinaryTree:
    def __init__(self, children_left: np.array, children_right: np.array, children_default: np.array,
                 split_features_index: np.array, split_values: np.array,
                 values: np.array, train_node_weights: np.array):
        self.children_left = children_left
        self.children_right = children_right
        self.children_default = children_default
        self.split_features_index = split_features_index
        self.split_values = split_values
        self.values = values
        self.train_node_weights = train_node_weights  # does it really correspond to train samples ? (or train + valid ?) I think pbm with bootstrap also
        self.n_leaves = len(train_node_weights) - len(self.children_left)

    def compute_feature_contribs(self, node_weights1: np.array, node_weights2: np.array,
                                     n_features: int) -> dict:
        """

        :param node_weights1:
        :param node_weights2: Usually correspond to train dataset
        :param n_features:
        :return:
        """
        assert len(node_weights1) == len(node_weights2)
        split_contribs = self._compute_split_contribs(node_weights1, node_weights2)
        feature_contribs = np.zeros((n_features, split_contribs.shape[1]))
        for feature_index, contribs in zip(self.split_features_index, split_contribs):
            feature_contribs[feature_index, :] += contribs
        return feature_contribs

    def _compute_split_contribs(self, node_weights1: np.array, node_weights2: np.array):
        split_drifts1 = self._compute_split_drifts(node_weights1)
        split_drifts2 = self._compute_split_drifts(node_weights2)
        split_contribs = split_drifts2 - split_drifts1
        return split_contribs

    def _compute_split_drifts(self, node_weights: np.array) -> np.array:
        """
        :param node_weights:
        :return:
        """
        n_nodes = len(self.children_left)
        sample_weight_fractions = node_weights / node_weights[0]
        split_drifts = np.zeros((n_nodes, self.values.shape[1]))
        for i in range(n_nodes):
            split_drifts[i] = (sample_weight_fractions[self.children_left[i]] * self.values[self.children_left[i]] +
                                sample_weight_fractions[self.children_right[i]] * self.values[self.children_right[i]] -
                                sample_weight_fractions[i] * self.values[i])
        return split_drifts
