from .i_drift_corrector import IDriftCorrector
from ..model_parser.i_tree_ensemble_parser import ITreeEnsembleParser
from ..model_parser.single_tree import BinaryTree
import numpy as np
from ..common.math_utils import threshold


class TreeEnsembleDriftCorrector(IDriftCorrector):

    def __init__(self, model_parser: ITreeEnsembleParser, X1, max_depth: int, max_ratio: int):
        self.model_parser = model_parser
        self.X1 = X1
        self.max_depth = max_depth
        self.max_ratio = max_ratio

    def get_weights(self, return_object: bool):
        return self._get_weights(self.model_parser, self.X1, self.max_depth, self.max_ratio, return_object)

    def _get_weights(self, model_parser: ITreeEnsembleParser, X1, max_depth: int, max_ratio: int, return_object: bool):
        weights_all = np.zeros((X1.shape[0], model_parser.n_trees))
        predicted_leaves1 = model_parser.predict_leaf(X1)
        for i, tree in enumerate(model_parser.trees):
            weights_all[:, i] = self._get_weights_tree(tree, predicted_leaves1[:, i], model_parser.node_weights1[i],
                                                       model_parser.node_weights2[i], max_depth, max_ratio)
        geometric_mean_weights = np.power(weights_all.prod(axis=1), 1/model_parser.n_trees)
        if return_object:
            return geometric_mean_weights, self
        else:
            return geometric_mean_weights

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
