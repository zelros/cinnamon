import numpy as np
import pandas as pd
import json
from .single_tree import BinaryTree
import catboost
import tempfile
from typing import Tuple
from .abstract_tree_ensemble_parser import AbstractTreeEnsembleParser
from ..common.math_utils import reverse_binary_representation


class CatBoostParser(AbstractTreeEnsembleParser):
    objective_task_map = {'RMSE': 'regression',
                          'Logloss':'binary_classification',
                          'CrossEntropy': 'binary_classification',  # TODO: make sure it predicts logits
                          'MultiClass': 'multiclass_classification',
                          }

    not_supported_objective = {}

    def __init__(self, model, model_type, iteration_range, task: str):
        super().__init__(model, model_type, iteration_range)
        self.task = task

    def parse(self, iteration_range: Tuple[int, int] = None):
        tmp_file = tempfile.NamedTemporaryFile()
        self.original_model.save_model(tmp_file.name, format="json")
        self.json_cb_model = json.load(open(tmp_file.name, "r"))
        tmp_file.close()

        self.original_model_total_iterations = len(self.json_cb_model['oblivious_trees'])
        self.iteration_range = self._get_iteration_range(iteration_range, self.original_model_total_iterations)
        self.n_iterations = self.iteration_range[1] - self.iteration_range[0]  # corresponds to n trees after iteration_range

        # load the CatBoost oblivious trees specific parameters
        self.model_objective = self.json_cb_model['model_info']['params']['loss_function']['type']

        # I take the exact def of tree depth, so +1
        self.max_depth = self.json_cb_model['model_info']['params']['tree_learner_options']['depth']
        self.cat_feature_indices = self.original_model.get_cat_feature_indices()
        self.feature_names = self.original_model.feature_names_
        self.n_features = len(self.feature_names)
        self.class_names = [str(x) for x in self.original_model.classes_.tolist()]
        if self.class_names is not None and len(self.class_names) > 2:  # if multiclass
            self.prediction_dim = len(self.class_names)
        else:
            self.prediction_dim = 1
        self.base_score = self.json_cb_model['scale_and_bias'][1]  # first element of the list is scale

        # details for cat_features
        if len(self.cat_feature_indices) > 0:
            self.cat_feature_index_map = {feat['feature_index']: feat['flat_feature_index']
                                          for feat in self.json_cb_model['features_info']['categorical_features']}
            self.inv_cat_feature_index_map = {v: k for k, v in self.cat_feature_index_map.items()}
            self.cat_label_map = {feat['feature_index']: feat['values'] if 'values' in feat else None
                                  for feat in self.json_cb_model['features_info']['categorical_features']}

        self.n_trees = self.n_iterations
        self.trees = self._get_trees(self.json_cb_model, self.iteration_range, self.n_features)

    def _get_trees(self, json_cb_model, iteration_range, n_features):
        # load all trees
        trees = []
        for tree_index in range(iteration_range[0], iteration_range[1]):
            # leaf weights
            leaf_weights = json_cb_model['oblivious_trees'][tree_index]['leaf_weights']
            # ------------------------------------------------------------------------------------------------
            # BugFix : correcting the way CatBoost numbers the predicted leaves (with order of splits inversed)
            n_bits = int(np.log2(len(leaf_weights)))
            mapping = [reverse_binary_representation(x, n_bits) for x in range(2**n_bits)]
            # ------------------------------------------------------------------------------------------------

            leaf_weights = [leaf_weights[mapping[i]] for i in range(len(leaf_weights))]
            # re-compute the number of samples that pass through each node
            leaf_weights_unraveled = [0] * (len(leaf_weights) - 1) + leaf_weights
            for index in range(len(leaf_weights) - 2, -1, -1):
                leaf_weights_unraveled[index] = leaf_weights_unraveled[2 * index + 1] + leaf_weights_unraveled[2 * index + 2]

            # leaf values
            # leaf values = log odd if binary classification
            # leaf values = log softmax if multiclass classification
            leaf_values = json_cb_model['oblivious_trees'][tree_index]['leaf_values']
            n_class = int(len(leaf_values) / len(leaf_weights))

            # re-compute leaf values within each node
            leaf_values_unraveled = np.concatenate((np.zeros((len(leaf_weights) - 1, n_class)),
                                                    np.array(leaf_values).reshape(len(leaf_weights), n_class)[mapping]),
                                                   axis=0)
            for index in range(len(leaf_weights) - 2, -1, -1):
                if leaf_weights_unraveled[2 * index + 1] + leaf_weights_unraveled[2 * index + 2] == 0:
                    leaf_values_unraveled[index, :] = [-1] * n_class  # equal probabilities
                else:
                    leaf_values_unraveled[index, :] = \
                        (leaf_weights_unraveled[2 * index + 1] * leaf_values_unraveled[2 * index + 1, :] +
                         leaf_weights_unraveled[2 * index + 2] * leaf_values_unraveled[2 * index + 2, :]) / \
                        (leaf_weights_unraveled[2 * index + 1] + leaf_weights_unraveled[2 * index + 2])

            children_left = [i * 2 + 1 for i in range(len(leaf_weights) - 1)]
            children_left += [-1] * len(leaf_weights)

            children_right = [i * 2 for i in range(1, len(leaf_weights))]
            children_right += [-1] * len(leaf_weights)

            children_default = [i * 2 + 1 for i in range(len(leaf_weights) - 1)]
            children_default += [-1] * len(leaf_weights)

            # load the split features and borders
            # split features and borders go from leafs to the root
            split_features_index = []
            borders = []
            split_types = []
            for elem in json_cb_model['oblivious_trees'][tree_index]['splits']:
                split_type = elem.get('split_type')
                split_types.append(split_type)
                if split_type == 'FloatFeature':
                    split_features_index.append(json_cb_model['features_info']['float_features'][elem.get('float_feature_index')]['flat_feature_index'])
                    borders.append(elem['border'])
                elif split_type == 'OneHotFeature':
                    split_features_index.append(json_cb_model['features_info']['categorical_features'][elem.get('cat_feature_index')]['flat_feature_index'])
                    borders.append(elem['value'])
                elif split_type == 'OnlineCtr':
                    # TODO: only the first feature in the list is considered here (multiple feature
                    corresponding_cat_index = json_cb_model['features_info']['ctrs'][elem.get('ctr_target_border_idx')]['elements'][0]['cat_feature_index']
                    split_features_index.append(self.cat_feature_index_map[corresponding_cat_index])
                    borders.append(elem['border'])
                else:
                    self._model_parser_error()

            split_types_unraveled = []
            for counter, split_type in enumerate(split_types):  # go from root to the leaves
                split_types_unraveled += [split_type] * (2 ** counter)
            split_types_unraveled += [-1] * len(leaf_weights)

            split_features_index_unraveled = []
            for counter, feature_index in enumerate(split_features_index):
                split_features_index_unraveled += [feature_index] * (2 ** counter)
            split_features_index_unraveled += [-1] * len(leaf_weights)

            borders_unraveled = []
            for counter, border in enumerate(borders):
                borders_unraveled += [border] * (2 ** counter)
            borders_unraveled += [-1] * len(leaf_weights)

            trees.append(BinaryTree(children_left=np.array(children_left),
                                    children_right=np.array(children_right),
                                    children_default=np.array(children_default),
                                    split_features_index=np.array(split_features_index_unraveled),
                                    split_values=np.array(borders_unraveled),
                                    values=leaf_values_unraveled,
                                    train_node_weights=np.array(leaf_weights_unraveled),
                                    n_features=n_features,
                                    split_types=np.array(split_types_unraveled),
                                    ))
        return trees

    def predict_leaf(self, X: pd.DataFrame):
        # transform X into catboost.Pool
        pool = catboost.Pool(X, cat_features=[self.feature_names[i] for i in
                                              self.original_model.get_cat_feature_indices()])
        predicted_leaves = self.original_model.calc_leaf_indexes(pool,
                                                     ntree_start=self.iteration_range[0],
                                                     ntree_end=self.iteration_range[1])
        # ------------------------------------------------------------------------------------------------
        # BugFix : correcting the way CatBoost numbers the predicted leaves (with order of splits inversed)
        for i in range(predicted_leaves.shape[1]):
            n_leaves = int((self.trees[i].n_nodes + 1)/2)
            n_bits = int(np.log2(n_leaves))
            # numerical optimization: precompute all values
            mapping = [reverse_binary_representation(x, n_bits) for x in range(2**n_bits)]
            predicted_leaves[:, i] = np.array([mapping[x] for x in predicted_leaves[:, i]])
        # ------------------------------------------------------------------------------------------------

        # By default catboost numbers leaves from 0 to n_leaves - 1, hence we need to add the number of nodes
        # The number we had may be different depending on the depth of the tree
        for i in range(predicted_leaves.shape[1]):
            # this assumes that trees are perfect binary trees (which is default in CatBoost)
            predicted_leaves[:, i] += int((self.trees[i].n_nodes - 1)/2)
        return predicted_leaves

    def predict_raw(self, X: pd.DataFrame):
        pool = catboost.Pool(X, cat_features=self.cat_feature_indices)
        return self.original_model.predict(pool, prediction_type='RawFormulaVal',
                                               ntree_start=self.iteration_range[0],
                                               ntree_end=self.iteration_range[1])

    def predict_proba(self, X: pd.DataFrame):
        pool = catboost.Pool(X, cat_features=self.cat_feature_indices)
        if self.prediction_dim == 1:  # binary classif
            return self.original_model.predict_proba(pool, ntree_start=self.iteration_range[0],
                                                     ntree_end=self.iteration_range[1])[:, 1]
        else:  # multiclass
            return self.original_model.predict_proba(pool, ntree_start=self.iteration_range[0],
                                                     ntree_end=self.iteration_range[1])

    def predict_leaf_with_model_parser(self, X):
        # TODO: common - should be put in abstract class (but pbm there is a
        # a small difference between XGBoostParser and CatboostParser
        def down(node_idx: int, i: int, tree: BinaryTree) -> int:
            '''
            Recursive function to get leaf of a given observation
            :param node_idx:
            :param i: raw index of the observation in dataset X
            :param tree:
            '''
            if tree.children_left[node_idx] == -1:
                return node_idx
            else:
                col_idx = tree.split_features_index[node_idx]
                obs_value = X.iloc[i, col_idx]
                split_type = tree.split_types[node_idx]
                split_value = tree.split_values[node_idx]
                if split_type == 'OneHotFeature':
                    # at predict catboost compute hash for labels of cat features
                    hash = self.model_objective._object._calc_cat_feature_perfect_hash(obs_value, col_idx)
                    cat_feature_idx = self.inv_cat_feature_index_map[col_idx]
                    if self.cat_label_map[cat_feature_idx][hash] == split_value:
                        return down(tree.children_right[node_idx], i, tree)
                    else:
                        return down(tree.children_left[node_idx], i, tree)
                elif split_type == 'FloatFeature':
                    if obs_value > tree.split_values[node_idx]:
                        return down(tree.children_right[node_idx], i, tree)
                    else:
                        return down(tree.children_left[node_idx], i, tree)
                elif split_type == 'OnlineCtr':
                    # FIXME: handle OnlineCtr for Catboost
                    self._model_parser_error()
                else:
                    self._model_parser_error()

        leaf_indexes = []
        for i in range(X.shape[0]):
            leaf_indexes_obs = []
            for tree_idx in range(self.n_trees):
                leaf_indexes_obs.append(down(0, i, self.trees[tree_idx]))
            leaf_indexes.append(leaf_indexes_obs)
        return np.array(leaf_indexes, dtype=np.int32)
