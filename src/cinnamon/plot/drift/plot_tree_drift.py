from treelib import Tree
from typing import List

from ...drift.model_drift_explainer import ModelDriftExplainer
from ...common.constants import TreeBasedDriftValueType
from ...model_parser.abstract_tree_ensemble_parser import AbstractTreeEnsembleParser
from ...model_parser.single_tree import BinaryTree


def plot_tree_drift(
        drift_explainer: ModelDriftExplainer,
        tree_idx: int,
        type: str = TreeBasedDriftValueType.MEAN.value) -> None:
    """
    Plot the representation of a given tree in the model, to illustrate how
    drift importances are computed.

    See the documentation in README for explanations about how it is computed,
    especially the slide presentation.

    Parameters
    ----------
    drift_explainer: ModelDriftExplainer
        A ModelDriftExplainer object.

    tree_idx : int
        Index of the tree to plot

    type: str, optional (default="mean")
        Method used for drift importances computation.
        Choose among:
        - "node_size"
        - "mean"
        - "mean_norm"

        See details in slide presentation.

    Returns
    -------
    None
    """
    _plot_tree_drift(drift_explainer._model_parser, tree_idx,
                     type, drift_explainer.feature_names)


def _plot_tree_drift(model_parser: AbstractTreeEnsembleParser, tree_idx: int, type: str, feature_names: List[str]) -> None:
    if model_parser.node_weights1 is None:
        raise ValueError(
            'You need to run drift_explainer.fit before calling plot_tree_drift')
    if type not in [e.value for e in TreeBasedDriftValueType]:
        raise ValueError(f'Bad value for "type"')
    else:
        plot_drift(binary_tree=model_parser.trees[tree_idx],
                   node_weights1=model_parser.node_weights1[tree_idx],
                   node_weights2=model_parser.node_weights2[tree_idx],
                   type=type,
                   feature_names=feature_names)


def plot_drift(binary_tree: BinaryTree, node_weights1, node_weights2, type, feature_names):
    split_contribs = binary_tree._compute_split_contribs(
        node_weights1, node_weights2, type)
    node_weight_fractions1 = node_weights1 / node_weights1[0]
    node_weight_fractions2 = node_weights2 / node_weights2[0]
    tree = Tree()
    for i in range(binary_tree.n_nodes):
        if i == 0:
            parent = None
        else:
            parent = binary_tree.get_parent(i)
        if node_weight_fractions1[i] != 0 or node_weight_fractions2[i] != 0:
            if binary_tree.children_left[i] == -1:
                tag = f'({round(node_weight_fractions1[i], 3)}, {round(node_weight_fractions2[i], 3)})'
            else:
                tag = f'{feature_names[binary_tree.split_features_index[i]]} ' \
                    f'({round(node_weight_fractions1[i], 3)}, {round(node_weight_fractions2[i], 3)}) - ' \
                    f'{[round(x, 3) for x in split_contribs[i, :]]}'
            tree.create_node(tag=tag, identifier=i, parent=parent)
    tree.show()
