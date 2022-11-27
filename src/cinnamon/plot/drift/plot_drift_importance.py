import matplotlib.pyplot as plt
import numpy as np
from typing import List

from ...common.constants import ModelAgnosticDriftValueType, TreeBasedDriftValueType
from ...drift.adversarial_drift_explainer import AdversarialDriftExplainer
from ...drift.abstract_drift_explainer import AbstractDriftExplainer
from ...drift.model_drift_explainer import ModelDriftExplainer


def plot_adversarial_drift_values(adversarial_drift_explainer: AdversarialDriftExplainer, n: int = 10):
    """
    Plot drift values computed using the adversarial method. Here the drift values
    correspond to the means of the feature importance taken over the n_splits
    cross-validated adversarial classifiers.

    See the documentation in README for explanations about how it is computed,
    especially the slide presentation.

    Parameters
    ----------
    n : interger, optional (default=10)
        Top n features to represent in the plot.

    Returns
    -------
    None
    """
    drift_values = adversarial_drift_explainer.get_adversarial_drift_values()
    _plot_drift_values(adversarial_drift_explainer, drift_values,
                       n, adversarial_drift_explainer.feature_subset)


def plot_tree_based_drift_values(drift_explainer: ModelDriftExplainer, n: int = 10,
                                 type: str = TreeBasedDriftValueType.MEAN.value) -> None:
    """
    Plot drift values computed using the tree structures present in the model.

    See the documentation in README for explanations about how it is computed,
    especially the slide presentation.

    Parameters
    ----------
    n : int, optional (default=10)
        Top n features to represent in the plot.

    type: str, optional (default="node_size")
        Method used for drift values computation.
        Choose among:
        - "node_size" (recommended)
        - "mean"
        - "mean_norm"

        See details in slide presentation.

    Returns
    -------
    None
    """
    if drift_explainer._model_parser is None:
        raise ValueError(
            'You need to run drift_explainer.fit before you can plot drift_values')

    drift_values = drift_explainer.get_tree_based_drift_values(type=type)
    _plot_drift_values(drift_explainer, drift_values,
                       n, drift_explainer.feature_names)


def plot_model_agnostic_drift_values(drift_explainer: ModelDriftExplainer, n: int = 10,
                                     type: str = ModelAgnosticDriftValueType.MEAN.value, prediction_type: str = "raw",
                                     max_ratio: float = 10, max_n_cat: int = 20) -> None:
    if type not in [x.value for x in ModelAgnosticDriftValueType]:
        raise ValueError(f'Bad value for "type": {type}')
    drift_values = drift_explainer.get_model_agnostic_drift_values(
        type, prediction_type, max_ratio, max_n_cat)
    _plot_drift_values(drift_explainer, drift_values,
                       n, drift_explainer.feature_names)


def _plot_drift_values(drift_explainer: AbstractDriftExplainer, drift_values: np.array,
                       n: int, feature_names: List[str]):
    # threshold n if n > drift_values.shape[0]
    n = min(n, drift_values.shape[0])

    # sort by importance in terms of drift
    # sort in decreasing order according to sum of absolute values of drift_values
    order = np.abs(drift_values).sum(axis=1).argsort()[::-1].tolist()
    ordered_names = [feature_names[i] for i in order]
    ordered_drift_values = drift_values[order, :]

    n_dim = drift_values.shape[1]
    legend_labels = drift_explainer.class_names if n_dim > 1 else []

    # plot
    fig, ax = plt.subplots(figsize=(10, 10))
    X = np.arange(n)
    for i in range(n_dim):
        ax.barh(X + (n_dim-i-1)/(n_dim+1),
                ordered_drift_values[:n, i][::-1], height=1/(n_dim+1))
    ax.legend(legend_labels)
    ax.set_yticks(X + 1/(n_dim+1) * (n_dim-1)/2)
    ax.set_yticklabels(ordered_names[:n][::-1], fontsize=15)
    ax.set_xlabel('Drift values per feature', fontsize=15)
    plt.show()
