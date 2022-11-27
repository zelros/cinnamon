import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np
from ...drift.drift_utils import compute_distributions_cat, compute_distributions_num


def plot_drift_cat(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None, title=None,
                   max_n_cat: int = None, figsize=(10, 6),
                   legend_labels: Tuple[str, str] = ('Dataset 1', 'Dataset 2')) -> None:
    # compute both distributions
    distrib = compute_distributions_cat(
        a1, a2, sample_weights1, sample_weights2, max_n_cat)
    # len(distrib) rows and 2 columns
    bar_height = np.array([v for v in distrib.values()])

    # plot
    index = np.arange(len(distrib))
    bar_width = 0.35
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(index, bar_height[:, 0], bar_width)
    ax.bar(index + bar_width, bar_height[:, 1], bar_width)

    ax.set_xlabel('Category')
    ax.set_ylabel('Percentage')
    ax.set_title(title)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(list(distrib.keys()), rotation=30)
    ax.legend(legend_labels)
    plt.show()


def plot_drift_num(a1: np.array, a2: np.array, sample_weights1: np.array = None, sample_weights2: np.array = None,
                   title=None, figsize=(7, 5), bins=10,
                   legend_labels: Tuple[str, str] = ('Dataset 1', 'Dataset 2')) -> None:
    distribs = compute_distributions_num(
        a1, a2, bins, sample_weights1, sample_weights2)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(a1, bins=distribs['bin_edges'],
            density=True, weights=sample_weights1, alpha=0.3)
    ax.hist(a2, bins=distribs['bin_edges'],
            density=True, weights=sample_weights2, alpha=0.3)
    ax.legend(legend_labels)
    plt.title(title)
    plt.show()


