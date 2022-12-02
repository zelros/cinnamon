import numpy as np
from treelib import Tree
from ..common.constants import TreeBasedDriftValueType

class BinaryTree:
    def __init__(self,
                 children_left: np.array,
                 children_right: np.array,
                 children_default: np.array,
                 split_features_index: np.array,
                 split_values: np.array,
                 values: np.array,
                 train_node_weights: np.array,
                 n_features: int,
                 split_types: np.array = None,  # only used in Catboost
                 ):
        self.children_left = children_left
        self.children_right = children_right
        self.children_default = children_default
        self.split_features_index = split_features_index
        self.split_values = split_values
        self.values = values
        self.train_node_weights = train_node_weights  # does it really correspond to train samples ? (or train + valid ?) I think pbm with bootstrap also
        self.n_features = n_features
        self.n_nodes = len(self.children_left)
        self.split_types = split_types

    def compute_drift_values(self, node_weights1: np.array, node_weights2: np.array, type: str) -> np.array:
        """
        :param node_weights1:
        :param node_weights2: Usually correspond to train dataset
        :return:
        """
        assert len(node_weights1) == len(node_weights2)
        split_contribs = self._compute_split_contribs(node_weights1, node_weights2, type)
        drift_values = np.zeros((self.n_features, split_contribs.shape[1]))
        for feature_index, contribs in zip(self.split_features_index, split_contribs):
            drift_values[feature_index, :] += contribs
        return drift_values

    def _compute_split_contribs(self, node_weights1: np.array, node_weights2: np.array, type: str):
        node_weight_fractions1 = node_weights1 / node_weights1[0]
        node_weight_fractions2 = node_weights2 / node_weights2[0]
        if type == TreeBasedDriftValueType.MEAN_NORM.value:
            return self._compute_split_contribs_mean_norm(node_weight_fractions1, node_weight_fractions2)
        elif type == TreeBasedDriftValueType.MEAN.value:
            return self._compute_split_contribs_mean(node_weight_fractions1, node_weight_fractions2)
        elif type == TreeBasedDriftValueType.NODE_SIZE.value:
            return self._compute_split_contribs_node_size(node_weight_fractions1, node_weight_fractions2)

    def _compute_split_contribs_mean(self, node_weight_fractions1, node_weight_fractions2):
        split_contribs = np.zeros((self.n_nodes, self.values.shape[1]))
        for i in range(self.n_nodes):
            if self.children_left[i] == -1:  # if leaf
                continue  # equal to 0 by default
            else:
                mean_log_softmax1 = (node_weight_fractions1[self.children_left[i]] * self.values[self.children_left[i]] +
                                     node_weight_fractions1[self.children_right[i]] * self.values[self.children_right[i]] -
                                     node_weight_fractions1[i] * self.values[i])
                mean_log_softmax2 = (node_weight_fractions2[self.children_left[i]] * self.values[self.children_left[i]] +
                                     node_weight_fractions2[self.children_right[i]] * self.values[self.children_right[i]] -
                                     node_weight_fractions2[i] * self.values[i])
                split_contribs[i] = mean_log_softmax2 - mean_log_softmax1
        return split_contribs

    def _compute_split_contribs_node_size(self, node_weight_fractions1, node_weight_fractions2):
        split_contribs = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            if self.children_left[i] == -1:  # if leaf
                split_contribs[i] = 0
            elif node_weight_fractions1[i] == 0 or node_weight_fractions2[i] == 0:
                split_contribs[i] = 0
            else:
                # the diff is weighted by the minimum of node_weight_fractions1[i] and node_weight_fractions2[i]
                split_contribs[i] = (abs(node_weight_fractions1[self.children_left[i]] / node_weight_fractions1[i] -
                                       node_weight_fractions2[self.children_left[i]] / node_weight_fractions2[i]) *
                                       min(node_weight_fractions1[i], node_weight_fractions2[i]))
        return split_contribs.reshape(len(split_contribs), 1)

    def _compute_split_contribs_mean_norm(self, node_weight_fractions1, node_weight_fractions2):
        split_contribs = np.zeros((self.n_nodes, self.values.shape[1]))
        for i in range(self.n_nodes):
            if self.children_left[i] == -1:  # if leaf
                continue  # equal to 0 by default
            if node_weight_fractions1[i] == 0 or node_weight_fractions2[i] == 0:
                continue
            else:
                mean_log_softmax1 = (node_weight_fractions1[self.children_left[i]] * self.values[self.children_left[i]] +
                                     node_weight_fractions1[self.children_right[i]] * self.values[self.children_right[i]] -
                                     node_weight_fractions1[i] * self.values[i]) / node_weight_fractions1[i]
                mean_log_softmax2 = (node_weight_fractions2[self.children_left[i]] * self.values[self.children_left[i]] +
                                     node_weight_fractions2[self.children_right[i]] * self.values[self.children_right[i]] -
                                     node_weight_fractions2[i] * self.values[i]) / node_weight_fractions2[i]
                # the diff of mean_log_softmax2 and mean_log_softmax1 is weighted by the minimum of
                # node_weight_fractions1[i] and node_weight_fractions2[i]
                split_contribs[i] = (mean_log_softmax2 - mean_log_softmax1) * min(node_weight_fractions1[i],
                                                                                  node_weight_fractions2[i])
        return split_contribs

    @staticmethod
    def _get_leaves(children_left):
        return [i for i in range(len(children_left)) if children_left[i] == -1]

    def plot_drift(self, node_weights1, node_weights2, type, feature_names):
        split_contribs = self._compute_split_contribs(node_weights1, node_weights2, type)
        node_weight_fractions1 = node_weights1 / node_weights1[0]
        node_weight_fractions2 = node_weights2 / node_weights2[0]
        tree = Tree()
        for i in range(self.n_nodes):
            if i == 0:
                parent = None
            else:
                parent = self.get_parent(i)
            if node_weight_fractions1[i] != 0 or node_weight_fractions2[i] != 0:
                if self.children_left[i] == -1:
                    tag = f'({round(node_weight_fractions1[i], 3)}, {round(node_weight_fractions2[i], 3)})'
                else:
                    tag = f'{feature_names[self.split_features_index[i]]} ' \
                          f'({round(node_weight_fractions1[i], 3)}, {round(node_weight_fractions2[i], 3)}) - ' \
                          f'{[round(x, 3) for x in split_contribs[i, :]]}'
                tree.create_node(tag=tag, identifier=i, parent=parent)
        tree.show()

    def get_parent(self, node_idx: int):
        if node_idx <= 0:
            raise ValueError(f'Bad value for node_idx: {node_idx}. node_idx should be > 0.')
        return np.where(self.children_left == node_idx)[0][0] if node_idx in self.children_left \
            else np.where(self.children_right == node_idx)[0][0]

    def get_depth(self, node_idx: int):
        if node_idx == 0:
            return 0
        else:
            return self.get_depth(self.get_parent(node_idx)) + 1

    def up(self, node_idx: int, n: int = 1):
        for _ in range(n):
            node_idx = self.get_parent(node_idx)
        return node_idx
