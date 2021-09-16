import numpy as np
from treelib import Node, Tree


class BinaryTree:
    def __init__(self, children_left: np.array, children_right: np.array, children_default: np.array,
                 split_features_index: np.array, split_values: np.array,
                 values: np.array, train_node_weights: np.array, n_features: int):
        self.children_left = children_left
        self.children_right = children_right
        self.children_default = children_default
        self.split_features_index = split_features_index
        self.split_values = split_values
        self.values = values
        self.train_node_weights = train_node_weights  # does it really correspond to train samples ? (or train + valid ?) I think pbm with bootstrap also
        self.n_features = n_features
        self.n_nodes = len(self.children_left)

    '''
    def plot(self, node_weights, feature_names=None):
        if feature_names is None:
            feature_names = [f'{i}' for i in range(self.n_features)]
        tree = Tree()
        for i in range(self.n_nodes):
            parent = self.children_left.index
            tree.create_node(tag=, identifier=i, parent=)
        pass
    '''

    def compute_feature_contribs(self, node_weights1: np.array, node_weights2: np.array, type: str) -> np.array:
        """
        :param node_weights1:
        :param node_weights2: Usually correspond to train dataset
        :return:
        """
        assert len(node_weights1) == len(node_weights2)
        split_contribs = self._compute_split_contribs(node_weights1, node_weights2, type)
        feature_contribs = np.zeros((self.n_features, split_contribs.shape[1]))
        for feature_index, contribs in zip(self.split_features_index, split_contribs):
            feature_contribs[feature_index, :] += contribs
        return feature_contribs

    def _compute_split_contribs(self, node_weights1: np.array, node_weights2: np.array, type: str):
        if type == 'mean_diff':
            return self._compute_split_mean_diffs(node_weights1, node_weights2)
        elif type == 'size_diff':
            return self._compute_split_size_diffs(node_weights1, node_weights2)

    def _compute_split_size_diffs(self, node_weights1, node_weights2):
        sample_weight_fractions1 = node_weights1 / node_weights1[0]
        sample_weight_fractions2 = node_weights2 / node_weights2[0]
        split_size_diffs = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            if self.children_left[i] == -1:  # if leaf
                split_size_diffs[i] = 0
            elif sample_weight_fractions1[i] == 0 or sample_weight_fractions2[i] == 0:
                split_size_diffs[i] = 0
            else:
                # the diff is weighted by the minimum of sample_weight_fractions1[i] and sample_weight_fractions2[i]
                split_size_diffs[i] = (abs(sample_weight_fractions1[self.children_left[i]] / sample_weight_fractions1[i] -
                                       sample_weight_fractions2[self.children_left[i]] / sample_weight_fractions2[i]) *
                                       min(sample_weight_fractions1[i], sample_weight_fractions2[i]))
        return split_size_diffs.reshape(len(split_size_diffs), 1)

    def _compute_split_mean_diffs(self, node_weights1, node_weights2):
        sample_weight_fractions1 = node_weights1 / node_weights1[0]
        sample_weight_fractions2 = node_weights2 / node_weights2[0]
        split_mean_diffs = np.zeros((self.n_nodes, self.values.shape[1]))
        for i in range(self.n_nodes):
            if self.children_left[i] == -1:  # if leaf
                continue  # equal to 0 by default
            if sample_weight_fractions1[i] == 0 or sample_weight_fractions2[i] == 0:
                continue
            else:
                mean_log_softmax1 = (sample_weight_fractions1[self.children_left[i]] * self.values[self.children_left[i]] +
                                     sample_weight_fractions1[self.children_right[i]] * self.values[self.children_right[i]] -
                                     sample_weight_fractions1[i] * self.values[i]) / sample_weight_fractions1[i]
                mean_log_softmax2 = (sample_weight_fractions2[self.children_left[i]] * self.values[self.children_left[i]] +
                                     sample_weight_fractions2[self.children_right[i]] * self.values[self.children_right[i]] -
                                     sample_weight_fractions2[i] * self.values[i]) / sample_weight_fractions2[i]
                # the diff of mean_log_softmax2 and mean_log_softmax1 is weighted by the minimum of
                # sample_weight_fractions1[i] and sample_weight_fractions2[i]
                split_mean_diffs[i] = (mean_log_softmax2 - mean_log_softmax1) * min(sample_weight_fractions1[i],
                                                                                    sample_weight_fractions2[i])
        return split_mean_diffs

    @staticmethod
    def _get_leaves(children_left):
        return [i for i in range(len(children_left)) if children_left[i] == -1]
