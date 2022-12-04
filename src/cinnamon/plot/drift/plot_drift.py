from typing import Tuple, Union

from ...drift.abstract_drift_explainer import AbstractDriftExplainer
from ...drift.model_drift_explainer import ModelDriftExplainer
from .plot_utils import plot_drift_cat, plot_drift_num


def plot_target_drift(drift_explainer: AbstractDriftExplainer, max_n_cat: int = 20, figsize: Tuple[int, int] = (7, 5), bins = 10,
                        legend_labels: Tuple[str, str] = ('Dataset 1', 'Dataset 2')):
    """
    Plot distributions of labels in order to
    visualize a potential drift of the target labels.

    Parameters
    ----------
    drift_explainer: AbstractDriftExplainer
        A AbstractDriftExplainer object.

    max_n_cat : int (default=20)
        For multiclass classification only. Maximum number of classes to
        represent on the plot.

    bins : int or sequence of scalars or str, optional (default=10)
        For regression only. 'two_heads' corresponds to a number of bins which is the minimum of
        of the optimal number of bins for dataset 1 and dataset 2 taken separately.
        Other value of "bins" parameter passed to matplotlib.pyplot.hist function are also
        accepted.

    figsize : Tuple[int, int] (default=(7, 5))
        Graphic size passed to matplotlib.

    legend_labels : Tuple[str, str] (default=('Dataset 1', 'Dataset 2'))
        Legend labels used for dataset 1 and dataset 2

    Returns
    -------
    None
    """
    if drift_explainer.y1 is None or drift_explainer.y2 is None:
        raise ValueError('"y1" or "y2" argument was not passed to drift_explainer.fit method')
    if drift_explainer.task == 'classification':
        plot_drift_cat(drift_explainer.y1, drift_explainer.y2, drift_explainer.sample_weights1, drift_explainer.sample_weights2, title='target',
                        max_n_cat=max_n_cat, figsize=figsize, legend_labels=legend_labels)
    elif drift_explainer.task == 'regression':
        plot_drift_num(drift_explainer.y1, drift_explainer.y2, drift_explainer.sample_weights1, drift_explainer.sample_weights2, title='target',
                        figsize=figsize, bins=bins, legend_labels=legend_labels)


def plot_feature_drift(drift_explainer: AbstractDriftExplainer, feature: Union[int, str], max_n_cat: int = 20, figsize: Tuple[int, int]=(7, 5),
                        as_discrete: bool = False, bins = 10,
                        legend_labels: Tuple[str, str] = ('Dataset 1', 'Dataset 2')):
    """
    Plot distributions of a given feature in order to
    visualize a potential data drift of this feature.

    Parameters
    ----------
    drift_explainer: AbstractDriftExplainer
        A AbstractDriftExplainer object.

    feature : Union[int, str]
        Either the column index or the name of the feature.

    max_n_cat : int (default=20)
        Maximum number of classes to represent on the plot (used only for
        categorical feature (not supported currently) or
        if as_discrete == True

    bins : int or sequence of scalars or str, optional (default=10)
        For regression only. 'two_heads' corresponds to a number of bins which is the minimum of
        of the optimal number of bins for dataset 1 and dataset 2 taken separately.
        Other value of "bins" parameter passed to matplotlib.pyplot.hist function are also
        accepted.

    figsize : Tuple[int, int] (default=(7, 5))
        Graphic size passed to matplotlib

    as_discrete: bool (default=False)
        If a numerical feature is discrete (has few unique values), consider
        it discrete to make the plot.

    legend_labels : Tuple[str, str] (default=('Dataset 1', 'Dataset 2'))
        Legend labels used for dataset 1 and dataset 2

    Returns
    -------
    None
    """
    if drift_explainer.X1 is None:
        raise ValueError('You must call the fit method before calling "get_feature_drift"')
    feature_index, feature_name = drift_explainer._check_feature_param(feature, drift_explainer.feature_names)
    if feature_index in drift_explainer.cat_feature_indices or as_discrete:
        plot_drift_cat(drift_explainer.X1.iloc[:,feature_index].values, drift_explainer.X2.iloc[:,feature_index].values,
                        drift_explainer.sample_weights1, drift_explainer.sample_weights2, title=feature_name, max_n_cat=max_n_cat,
                        figsize=figsize, legend_labels=legend_labels)
    else:
        plot_drift_num(drift_explainer.X1.iloc[:,feature_index].values, drift_explainer.X2.iloc[:,feature_index].values,
                        drift_explainer.sample_weights1, drift_explainer.sample_weights2, title=feature_name, figsize=figsize, bins=bins,
                        legend_labels=legend_labels)


def plot_prediction_drift(drift_explainer: ModelDriftExplainer, prediction_type='raw', bins = 10,
                            figsize: Tuple[int, int] = (7, 5),
                            legend_labels: Tuple[str, str] = ('Dataset 1', 'Dataset 2')) -> None:
    """
    Plot histogram of distribution of predictions for dataset 1 and dataset 2
    in order to visualize a potential drift of the predicted values.
    See the documentation in README for explanations about how it is computed,
    especially the slide presentation.

    Parameters
    ----------
    drift_explainer: ModelDriftExplainer
        A ModelDriftExplainer object.
    
    prediction_type: str, optional (default="raw")
        Type of predictions to consider.
        Choose among:
        - "raw": logit predictions (binary classification), log-softmax predictions
        (multiclass classification), regular predictions (regression)
        - "proba": predicted probabilities (only for classification models)
        - "class": predicted classes (only for classification model)

    bins : int or sequence of scalars or str, optional (default=10)
        For regression only. 'two_heads' corresponds to a number of bins which is the minimum of
        of the optimal number of bins for dataset 1 and dataset 2 taken separately.
        Other value of "bins" parameter passed to matplotlib.pyplot.hist function are also
        accepted.

    figsize : Tuple[int, int], optional (default=(7, 5))
        Graphic size passed to matplotlib

    legend_labels : Tuple[str, str] (default=('Dataset 1', 'Dataset 2'))
        Legend labels used for dataset 1 and dataset 2

    Returns
    -------
    None
    """
    pred1, pred2 = drift_explainer._get_predictions(prediction_type)

    if drift_explainer.task == 'classification':
        if prediction_type == 'class':
            plot_drift_cat(pred1, pred2, drift_explainer.sample_weights1, drift_explainer.sample_weights2, title=f'Predictions',
                            max_n_cat=20, figsize=figsize, legend_labels=legend_labels)
        else: # prediction_type in ['raw', 'proba']
            if drift_explainer._prediction_dim == 1: # binary classif
                plot_drift_num(pred1, pred2, drift_explainer.sample_weights1, drift_explainer.sample_weights2, title=f'Predictions',
                        figsize=figsize, bins=bins, legend_labels=legend_labels)
            else: # multiclass classif
                for i in range(drift_explainer._prediction_dim):
                    plot_drift_num(pred1[:, i], pred2[:, i], drift_explainer.sample_weights1, drift_explainer.sample_weights2,
                                title=f'{drift_explainer.class_names[i]}', figsize=figsize, bins=bins, legend_labels=legend_labels)
    else:  # regression or ranking
        plot_drift_num(pred1, pred2, drift_explainer.sample_weights1, drift_explainer.sample_weights2, title=f'Predictions',
                        figsize=figsize, bins=bins, legend_labels=legend_labels)
