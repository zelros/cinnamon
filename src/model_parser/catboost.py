import numpy as np
import pandas as pd
import json
from .single_tree import BinaryTree
import catboost
import tempfile
from .i_tree_ensemble import ITreeEnsembleParser


class CatBoostParser(ITreeEnsembleParser):
    objective_task_map = {'RMSE': 'regression',
                          'Logloss':'binary_classification',
                          'CrossEntropy': 'binary_classification',  # TODO: make sure it predicts logits
                          'MultiClass': 'multiclass_classification',
                          }

    not_supported_objective = {}

    def __init__(self, model_type: str, task: str):
        super().__init__()
        self.model_type = model_type
        self.task = task
        self.feature_names = None  # specific to catboost
        self.class_names = None  # specific to catboost

    def parse(self, model, iteration_range):
        self.original_model = model
        tmp_file = tempfile.NamedTemporaryFile()
        self.original_model.save_model(tmp_file.name, format="json")
        self.json_cb_model = json.load(open(tmp_file.name, "r"))
        tmp_file.close()

        self.iteration_range = self._get_iteration_range(iteration_range, len(self.json_cb_model['oblivious_trees']))
        self.n_trees = self.iteration_range[1] - self.iteration_range[0]  # corresponds to n trees after iteration_range
        self.trees = self._get_trees(self.json_cb_model, self.iteration_range)

        # load the CatBoost oblivious trees specific parameters
        self.model_objective = self.json_cb_model['model_info']['params']['loss_function']['type']

        # I take the exact def of tree depth, so +1
        self.max_depth = self.json_cb_model['model_info']['params']['tree_learner_options']['depth'] + 1
        self.cat_feature_indices = self.original_model.get_cat_feature_indices()
        self.feature_names = self.original_model.feature_names_
        self.n_features = len(self.feature_names)
        self.class_names = self.original_model.classes_.tolist()
        if self.class_names is not None and len(self.class_names) > 2:  # if multiclass
            self.prediction_dim = len(self.class_names)
        else:
            self.prediction_dim = 1

    @staticmethod
    def _get_trees(json_cb_model, iteration_range):
        # load all trees
        trees = []
        for tree_index in range(iteration_range[0], iteration_range[1]):
            # leaf weights
            leaf_weights = json_cb_model['oblivious_trees'][tree_index]['leaf_weights']
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
                                                    np.array(leaf_values).reshape(len(leaf_weights), n_class)), axis=0)
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
            for elem in json_cb_model['oblivious_trees'][tree_index]['splits']:
                split_type = elem.get('split_type')
                if split_type == 'FloatFeature':
                    split_features_index.append(elem.get('float_feature_index'))
                    borders.append(elem['border'])
                elif split_type == 'OneHotFeature':
                    split_features_index.append(elem.get('cat_feature_index'))
                    borders.append(elem['value'])
                else:
                    split_features_index.append(elem.get('ctr_target_border_idx'))
                    borders.append(elem['border'])

            split_features_index_unraveled = []
            for counter, feature_index in enumerate(split_features_index[::-1]):  # go from leafs to the root
                split_features_index_unraveled += [feature_index] * (2 ** counter)
            split_features_index_unraveled += [-1] * len(leaf_weights)

            borders_unraveled = []
            for counter, border in enumerate(borders[::-1]):
                borders_unraveled += [border] * (2 ** counter)
            borders_unraveled += [-1] * len(leaf_weights)

            trees.append(BinaryTree(children_left=np.array(children_left),
                                    children_right=np.array(children_right),
                                    children_default=np.array(children_default),
                                    split_features_index=np.array(split_features_index_unraveled),
                                    split_values=np.array(borders_unraveled),
                                    values=leaf_values_unraveled,
                                    train_node_weights=np.array(leaf_weights_unraveled),
                                    ))
        return trees

    def predict_leaf(self, X: pd.DataFrame):
        # transform X into catboost.Pool
        pool = catboost.Pool(X, cat_features=[self.feature_names[i] for i in
                                              self.original_model.get_cat_feature_indices()])
        predicted_leaves = self.original_model.calc_leaf_indexes(pool,
                                                     ntree_start=self.iteration_range[0],
                                                     ntree_end=self.iteration_range[1])
        # By default catboost numbers leaves from 0 to n_leaves - 1, hence we need to add the number of nodes
        # The number we had may be different depending on the depth of the tree
        for i in range(predicted_leaves.shape[1]):
            # this assumes that trees are perfect binary trees (which is default inn CatBoost)
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
