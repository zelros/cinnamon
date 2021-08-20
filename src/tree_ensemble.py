from typing import List
import numpy as np
import pandas as pd
import json
from .single_tree import BinaryTree
import catboost


class CatBoostParser: # differentiate regressor and classifier ?
    def __init__(self, cb_model):
        import tempfile
        self.cb_model = cb_model
        tmp_file = tempfile.NamedTemporaryFile()
        cb_model.save_model(tmp_file.name, format="json")
        self.json_cb_model = json.load(open(tmp_file.name, "r"))
        tmp_file.close()

        # load the CatBoost oblivious trees specific parameters
        self.num_trees = len(self.json_cb_model['oblivious_trees'])
        self.max_depth = self.json_cb_model['model_info']['params']['tree_learner_options']['depth']
        self.feature_names = cb_model.feature_names_
        self.cat_features = [self.feature_names[i] for i in self.cb_model.get_cat_feature_indices()]
        self.n_features = len(self.feature_names)
        self.trees = self.get_trees()
        self.class_names = cb_model.classes_.tolist()
        self.model_objective = self._get_model_objective() # can I do without it ? it may cause bugs?..

    def get_trees(self):
        # load all trees
        trees = []
        for tree_index in range(self.num_trees):
            # leaf weights
            leaf_weights = self.json_cb_model['oblivious_trees'][tree_index]['leaf_weights']
            # re-compute the number of samples that pass through each node
            leaf_weights_unraveled = [0] * (len(leaf_weights) - 1) + leaf_weights
            leaf_weights_unraveled[0] = sum(leaf_weights)
            for index in range(len(leaf_weights) - 2, 0, -1):
                leaf_weights_unraveled[index] = leaf_weights_unraveled[2 * index + 1] + leaf_weights_unraveled[2 * index + 2]

            # leaf values
            # leaf values = log odd if binary classification
            # leaf values = log softmax if multiclass classification
            leaf_values = self.json_cb_model['oblivious_trees'][tree_index]['leaf_values']
            n_class = int(len(leaf_values) / len(leaf_weights))
            # re-compute leaf values within each node
            leaf_values_unraveled = np.concatenate((np.zeros((len(leaf_weights) - 1, n_class)),
                                                   np.array(leaf_values).reshape(len(leaf_weights), n_class)), axis=0)
            for index in range(len(leaf_weights) - 2, -1, -1):
                if leaf_weights_unraveled[2 * index + 1] + leaf_weights_unraveled[2 * index + 2] == 0:
                    leaf_values_unraveled[index, :] = [-1] * n_class
                else:
                    leaf_values_unraveled[index, :] = \
                        (leaf_weights_unraveled[2 * index + 1] * leaf_values_unraveled[2 * index + 1, :] +
                         leaf_weights_unraveled[2 * index + 2] * leaf_values_unraveled[2 * index + 2, :]) / \
                        (leaf_weights_unraveled[2 * index + 1] + leaf_weights_unraveled[2 * index + 2])

            children_left = [i * 2 + 1 for i in range(len(leaf_weights) - 1)]
            #children_left += [-1] * len(leaf_weights)

            children_right = [i * 2 for i in range(1, len(leaf_weights))]
            #children_right += [-1] * len(leaf_weights)

            children_default = [i * 2 + 1 for i in range(len(leaf_weights) - 1)]
            #children_default += [-1] * len(leaf_weights)

            # load the split features and borders
            # split features and borders go from leafs to the root
            split_features_index = []
            borders = []
            for elem in self.json_cb_model['oblivious_trees'][tree_index]['splits']:
                split_type = elem.get('split_type')
                if split_type == 'FloatFeature':
                    split_feature_index = elem.get('float_feature_index')
                    borders.append(elem['border'])
                elif split_type == 'OneHotFeature':
                    split_feature_index = elem.get('cat_feature_index')
                    borders.append(elem['value'])
                else:
                    split_feature_index = elem.get('ctr_target_border_idx')
                    borders.append(elem['border'])
                split_features_index.append(split_feature_index)

            split_features_index_unraveled = []
            for counter, feature_index in enumerate(split_features_index[::-1]):  # go from leafs to the root
                split_features_index_unraveled += [feature_index] * (2 ** counter)
            #split_features_index_unraveled += [0] * len(leaf_weights)

            borders_unraveled = []
            for counter, border in enumerate(borders[::-1]):
                borders_unraveled += [border] * (2 ** counter)
            #borders_unraveled += [0] * len(leaf_weights)

            trees.append(BinaryTree(children_left=np.array(children_left),
                                    children_right=np.array(children_right),
                                    children_default=np.array(children_default),
                                    split_features_index=np.array(split_features_index_unraveled),
                                    split_values=np.array(borders_unraveled),
                                    values=leaf_values_unraveled,
                                    train_node_weights=np.array(leaf_weights_unraveled),
                                    ))
        return trees

    def get_node_weights(self, X: pd.DataFrame, sample_weights: np.array) -> List[np.array]:
        """return sum of observation weights in each node of each tree of the model"""

        if sample_weights is None:
            sample_weights = np.ones(len(X))

        # transform X into catboost.Pool
        pool = catboost.Pool(X, cat_features=[self.feature_names[i] for i in self.cb_model.get_cat_feature_indices()])

        """pass X through the trees : compute node sample weights, and node values fro each tree"""
        leaf_indexes = self.cb_model.calc_leaf_indexes(pool) # voir si catboost.Pool ou bien dataframe etc.
        node_weights = []
        for index in range(self.num_trees):
            tree = self.trees[index]

            # compute sample weighs in leaves
            leaf_sample_weights_in_tree = [np.sum(sample_weights[leaf_indexes[:, index] == j], dtype=np.int32)
                                           for j in range(tree.n_leaves)]

            # add sample weights of nodes
            node_weights_in_tree = [0] * (len(leaf_sample_weights_in_tree) - 1) + leaf_sample_weights_in_tree
            for index in range(len(leaf_sample_weights_in_tree) - 2, -1, -1):
                node_weights_in_tree[index] = node_weights_in_tree[2 * index + 1] + node_weights_in_tree[2 * index + 2]

            # update node_weights
            node_weights.append(np.array(node_weights_in_tree, dtype=np.int32))
        return node_weights

    def get_predictions(self, X: pd.DataFrame) -> np.array:
        # array of shape (nb. obs, nb. class) for multiclass and shape array of shape (nb. obs, )
        # for binary class and regression
        pool = catboost.Pool(X, cat_features=[self.feature_names[i] for i in self.cb_model.get_cat_feature_indices()])
        return  self.cb_model.predict(pool, prediction_type='RawFormulaVal')

    def _get_model_objective(self):
        if self.json_cb_model['model_info']['params']['loss_function']['type'] in ['MultiClass']:
            return 'multiclass_classification'
