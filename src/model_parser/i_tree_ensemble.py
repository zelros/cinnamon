import pandas as pd
import numpy as np
from typing import List


class ITreeEnsembleParser:
    def __init__(self):
        self.model_type = None
        self.original_model = None
        self.n_trees = None
        self.max_depth = None
        self.cat_feature_indices = None
        self.n_features = None
        self.trees = None
        self.prediction_dim = None
        self.model_objective = None
        self.task = None
        self.iteration_range = None

        # specific to catboost
        self.class_names = None
        self.feature_names = None

    def parse(self, model, iteration_range):
        pass

    def predict_leaf(self, X: pd.DataFrame):
        pass

    def predict_raw(self, X: pd.DataFrame):
        pass

    def predict_proba(self, X: pd.DataFrame):
        pass

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
            ValueError(f'Bad value for prediction_type: {prediction_type}')

    def get_node_weights(self, X: pd.DataFrame, sample_weights: np.array) -> List[np.array]:
        """return sum of observation weights in each node of each tree of the model"""

        if sample_weights is None:
            sample_weights = np.ones(len(X))

        # pass X through the trees : compute node sample weights, and node values from each tree
        predicted_leaves = self.predict_leaf(X)
        node_weights = []
        for index in range(self.n_trees):
            tree = self.trees[index]

            # add sample weighs of terminal leaves
            node_weights_in_tree = []
            for j in range(tree.n_nodes):
                if tree.children_left[j] == -1:  # if leaf
                    node_weights_in_tree.append(np.sum(sample_weights[predicted_leaves[:, index] == j],
                                                       dtype=np.int32))
                else:  # if not a leaf
                    node_weights_in_tree.append(-1)

            # populate in reverse order to add sample weights of nodes
            for j in range(tree.n_nodes-1, -1, -1):
                if node_weights_in_tree[j] == -1:  # if not a leaf
                    node_weights_in_tree[j] = (node_weights_in_tree[tree.children_left[j]] +
                                               node_weights_in_tree[tree.children_right[j]])

            # update node_weights
            node_weights.append(np.array(node_weights_in_tree, dtype=np.int32))
        return node_weights

    @staticmethod
    def _get_iteration_range(iteration_range, initial_n_trees):
        if iteration_range is None:
            iteration_range = (0, initial_n_trees)
        elif iteration_range[1] > initial_n_trees:
            ValueError(f'"iteration_range" values exceeds {initial_n_trees} which is the number of trees in the model')
        else:
            pass
        return iteration_range
