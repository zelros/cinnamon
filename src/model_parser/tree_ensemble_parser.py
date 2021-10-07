from .i_tree_ensemble_parser import ITreeEnsembleParser
from typing import List
import pandas as pd
import numpy as np


class TreeEnsembleParser(ITreeEnsembleParser):

    def __init__(self, model, model_type, iteration_range, X):
        super().__init__()
        self.original_model = model
        self.model_type = model_type
        self.iteration_range = iteration_range
        self.parse(iteration_range, X)

    def fit(self, X1, X2, sample_weights1, sample_weights2):
        self.node_weights1 = self.get_node_weights(X1, sample_weights=sample_weights1)
        self.node_weights2 = self.get_node_weights(X2, sample_weights=sample_weights2)
        #self._check_feature_contribs_mean(X1, X2, sample_weights1, sample_weights2)

    # TODO: make abstract class instead of this interface
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
    def _get_iteration_range(iteration_range, initial_n_trees):
        if iteration_range is None:
            iteration_range = (0, initial_n_trees)
        elif iteration_range[1] > initial_n_trees:
            raise ValueError(f'"iteration_range" values exceeds {initial_n_trees} which is the number of trees in the model')
        else:
            pass
        return iteration_range

    @staticmethod
    def _model_parser_error():
        raise ValueError('Error in parsing "model": the passed model is not supported in DriftExplainer')

    def _check_parsing_with_leaf_predictions(self, X):
        if not np.array_equal(self.predict_leaf_with_model_parser(X), self.predict_leaf(X)):
            self._model_parser_error()

    def _check_feature_contribs_mean(self, X1, X2, sample_weights1, sample_weights2):
        sample_weights1_norm = sample_weights1 / np.sum(sample_weights1)
        sample_weights2_norm = sample_weights2 / np.sum(sample_weights2)
        if self.prediction_dim == 1:
            mean_prediction_diff = np.sum(sample_weights2_norm * self.predict_raw(X2)) - \
                                   np.sum(sample_weights1_norm * self.predict_raw(X1))
        else:
            mean_prediction_diff = np.sum(sample_weights2_norm[:, np.newaxis] * self.predict_raw(X2), axis=0) - \
                                   np.sum(sample_weights1_norm[:, np.newaxis] * self.predict_raw(X1), axis=0)
        stat = abs(self.compute_feature_contribs(type='mean').sum(axis=0) - mean_prediction_diff)
        if any(stat > 10**(-3)):  # any works because difference is an array
            raise ValueError('Error in computation of feature contributions')

    def compute_feature_contribs(self, type: str):
        """
        :param node_weights1:
        :param node_weights2:
        :param type: type: 'mean_norm', 'node_size', or 'wasserstein'
        :return:
        """
        if type == 'node_size':
            feature_contribs = np.zeros((self.n_features, 1))
        elif type in ['mean', 'mean_norm']:
            feature_contribs = np.zeros((self.n_features, self.prediction_dim))
        else:
            raise ValueError(f'Bad value for "type": {type}')

        feature_contribs_details = []
        for i, tree in enumerate(self.trees):
            feature_contribs_tree = tree.compute_feature_contribs(self.node_weights1[i], self.node_weights2[i],
                                                                  type=type)
            feature_contribs = self.add_feature_contribs(feature_contribs, feature_contribs_tree, i,
                                                         self.prediction_dim, type)
            feature_contribs_details.append(feature_contribs_tree)
        return feature_contribs  #, feature_contribs_details

    @staticmethod
    def add_feature_contribs(feature_contribs, feature_contribs_tree, i, prediction_dim, type):
        feature_contribs += feature_contribs_tree
        return feature_contribs
