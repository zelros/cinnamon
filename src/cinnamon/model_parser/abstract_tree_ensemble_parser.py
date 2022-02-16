from .i_tree_ensemble_parser import ITreeEnsembleParser
from typing import List, Tuple
import pandas as pd
import numpy as np

from .single_tree import BinaryTree
from ..common.math_utils import threshold
from ..common.constants import TreeBasedDriftValueType
from ..common.logging import cinnamon_logger


class AbstractTreeEnsembleParser(ITreeEnsembleParser):

    logger = cinnamon_logger.getChild('TreeEnsembleParser')

    def __init__(self, model, model_type, iteration_range):
        super().__init__()
        self.original_model = model
        self.model_type = model_type
        self.iteration_range = iteration_range
        self.trees = None
        self.parse(iteration_range)
        self.original_model_total_iterations = None

    def fit(self, X1, X2, sample_weights1, sample_weights2):
        self.node_weights1 = self.get_node_weights(X1, sample_weights=sample_weights1)
        self.node_weights2 = self.get_node_weights(X2, sample_weights=sample_weights2)

    def get_predictions(self, X: pd.DataFrame, prediction_type: str) -> np.array:
        # return array of shape (nb. obs, nb. class) for multiclass and shape array of shape (nb. obs, )
        # for binary class and regression
        """

        :param X:
        :param prediction_type: "raw" or "proba"
        :return:
        """
        # array of shape (nb. obs, nb. class) for multiclass and shape array of shape (nb. obs, )
        # for binary class and regression
        if prediction_type == 'raw':
            return self.predict_raw(X)
        elif prediction_type == 'proba':
            return self.predict_proba(X)
        else:
            raise ValueError(f'Bad value for prediction_type: {prediction_type}')

    def get_node_weights(self, X: pd.DataFrame, sample_weights: np.array) -> List[np.array]:
        """return sum of observation weights in each node of each tree of the model"""

        # pass X through the trees : compute node sample weights, and node values from each tree
        predicted_leaves = self.predict_leaf(X)
        node_weights = []
        for index in range(self.n_trees):
            tree = self.trees[index]

            # add sample weighs of terminal leaves
            node_weights_in_tree = []
            for j in range(tree.n_nodes):
                if tree.children_left[j] == -1:  # if leaf
                    node_weights_in_tree.append(np.sum(sample_weights[predicted_leaves[:, index] == j]))
                else:  # if not a leaf
                    node_weights_in_tree.append(-1)

            # populate in reverse order to add sample weights of nodes
            for j in range(tree.n_nodes-1, -1, -1):
                if node_weights_in_tree[j] == -1:  # if not a leaf
                    node_weights_in_tree[j] = (node_weights_in_tree[tree.children_left[j]] +
                                               node_weights_in_tree[tree.children_right[j]])

            # update node_weights
            node_weights.append(np.array(node_weights_in_tree))
        return node_weights

    @staticmethod
    def _get_iteration_range(iteration_range: Tuple[int, int], original_model_total_iterations: int,
                             best_iteration: int = None):
        if iteration_range is None:
            if best_iteration is not None:
                iteration_range = (0, best_iteration)
                AbstractTreeEnsembleParser.logger.warning('By default, the best iteration given by early stopping is used '
                                                          'to compute "iteration_range". This behavior is consistent with '
                                                          'model.predict XGBoost default behavior.')
            else:
                iteration_range = (0, original_model_total_iterations)
        elif iteration_range[1] > original_model_total_iterations:
            raise ValueError(f'"iteration_range" values exceeds the total number of trees in the model')
        else:
            pass
        return iteration_range

    @staticmethod
    def _model_parser_error():
        raise ValueError('Error in parsing "model": the passed model is not supported in DriftExplainer')

    def _check_parsing_with_leaf_predictions(self, X):
        if not np.array_equal(self.predict_leaf_with_model_parser(X), self.predict_leaf(X)):
            self._model_parser_error()

    def check_tree_based_drift_values_sum(self, X1, X2, sample_weights1, sample_weights2) -> None:
        sample_weights1_norm = sample_weights1 / np.sum(sample_weights1)
        sample_weights2_norm = sample_weights2 / np.sum(sample_weights2)
        if self.prediction_dim == 1:
            mean_prediction_diff = np.sum(sample_weights2_norm * self.predict_raw(X2)) - \
                                   np.sum(sample_weights1_norm * self.predict_raw(X1))
        else:
            mean_prediction_diff = np.sum(sample_weights2_norm[:, np.newaxis] * self.predict_raw(X2), axis=0) - \
                                   np.sum(sample_weights1_norm[:, np.newaxis] * self.predict_raw(X1), axis=0)
        stat = abs(self.compute_tree_based_drift_values(type=TreeBasedDriftValueType.MEAN.value).sum(axis=0) - mean_prediction_diff)
        if any(stat > 10**(-6)):  # any works because difference is an array
            raise ValueError('Error in computation of tree based drift values. Your model may not be properly parsed '
                             'by CinnaMon. You can report the error here: https://github.com/zelros/cinnamon/issues')

    def compute_tree_based_drift_values(self, type: str):
        """
        :param node_weights1:
        :param node_weights2:
        :param type: type: 'mean_norm', 'node_size', or 'mean'
        :return:
        """
        if type == TreeBasedDriftValueType.NODE_SIZE.value:
            drift_values = np.zeros((self.n_features, 1))
        elif type in [TreeBasedDriftValueType.MEAN.value,
                      TreeBasedDriftValueType.MEAN_NORM.value]:
            drift_values = np.zeros((self.n_features, self.prediction_dim))
        else:
            raise ValueError(f'Bad value for "type": {type}')

        drift_values_details = []
        for i, tree in enumerate(self.trees):
            drift_values_tree = tree.compute_drift_values(self.node_weights1[i], self.node_weights2[i],
                                                                  type=type)
            drift_values = self._add_drift_values(drift_values, drift_values_tree, i,
                                                         self.prediction_dim, type)
            drift_values_details.append(drift_values_tree)
        return drift_values  #, drift_values_details

    def plot_tree_drift(self, tree_idx: int, type: str, feature_names: List[str]) -> None:
        if self.node_weights1 is None:
            raise ValueError('You need to run drift_explainer.fit before calling plot_tree_drift')
        if type not in [e.value for e in TreeBasedDriftValueType]:
            raise ValueError(f'Bad value for "type"')
        else:
            self.trees[tree_idx].plot_drift(node_weights1=self.node_weights1[tree_idx],
                                                         node_weights2=self.node_weights2[tree_idx],
                                                         type=type,
                                                         feature_names=feature_names)

    def compute_tree_based_correction_weights(self, X1: pd.DataFrame, max_depth: int, max_ratio: int,
                                              sample_weights1: np.array) -> np.array:
        weights_all = np.zeros((X1.shape[0], self.n_trees))
        predicted_leaves1 = self.predict_leaf(X1)
        for i, tree in enumerate(self.trees):
            weights_all[:, i] = self._get_weights_tree(tree, predicted_leaves1[:, i], self.node_weights1[i],
                                                       self.node_weights2[i], max_depth, max_ratio)
        geometric_mean_weights = np.power(weights_all.prod(axis=1), 1/self.n_trees)  # corresponds to correction weights
        new_weights = sample_weights1 * geometric_mean_weights  # update initial sample_weights with the correction
        return new_weights * len(new_weights) / np.sum(new_weights)

    @staticmethod
    def _get_weights_tree(tree: BinaryTree, predicted_leaves: np.array, node_weights1: np.array,
                          node_weights2: np.array, max_depth: int, max_ratio: int):
        weights = np.zeros(len(predicted_leaves))
        for leaf_idx in np.unique(predicted_leaves):
            if max_depth is not None:
                leaf_depth = tree.get_depth(leaf_idx)
                if leaf_depth > max_depth:
                    # node_idx is the node above leaf that is at the good depth taking into account the max_depth
                    # parameter
                    node_idx = tree.up(leaf_idx, n=leaf_depth - max_depth)
                else:
                    node_idx = leaf_idx
            else:
                node_idx = leaf_idx
            node_weight_fractions1 = node_weights1 / node_weights1[0]
            node_weight_fractions2 = node_weights2 / node_weights2[0]
            # denominator can't be 0 because the leaf observation is inside the node
            weights[predicted_leaves == leaf_idx] = \
                threshold(node_weight_fractions2[node_idx] / node_weight_fractions1[node_idx], max_ratio)
        return weights

    @staticmethod
    def _add_drift_values(drift_values, drift_values_tree, i, prediction_dim, type):
        drift_values += drift_values_tree
        return drift_values

    def predict_leaf(self, X: pd.DataFrame) -> np.array:  # output dtype = np.int32
        pass

    def predict_raw(self, X: pd.DataFrame) -> np.array:
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        pass

    def predict_leaf_with_model_parser(self, X: pd.DataFrame) -> np.array:
        pass
