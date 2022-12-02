import numpy as np
from .drift_utils import AbstractDriftMetrics, compute_drift_num, compute_drift_cat, plot_drift_num, plot_drift_cat
from typing import List, Tuple
from ..common.logging import cinnamon_logger
from ..common.dev_utils import find_uniques
from ..common.stat_utils import (compute_regression_metrics, compute_classification_metrics,
                                 RegressionMetrics, ClassificationMetrics)
from .drift_utils import PerformanceMetricsDrift


class OutputDriftDetector:

    logger = cinnamon_logger.getChild('OutputDriftDetector')

    def __init__(self, task: str, prediction_type: str = None, class_names: List[str] = None):
        '''
        :param task: 'regression' or 'classification'
        :param prediction_type: for task=classification only ('raw', 'proba', or 'label'). label for the case where the user provides
        predicted labels and not probabilities, logit or log softmax. In the regression mode, it will be set to "raw" by default.
        :param class_names: leave None if regression. If classification, labels will based on "y" and "predictions"
        passed to fit, and sorted in alphanumeric order.
        '''
        self.predictions1 = None
        self.predictions2 = None
        self.y1 = None
        self.y2 = None
        self.sample_weights1 = None
        self.sample_weights2 = None
        self._prediction_dim = None
        self.task = self._check_input_task(task)
        self.prediction_type = self._check_input_prediction_type(prediction_type, self.task)
        self.class_names = self._check_input_class_names(class_names)

    def fit(self, predictions1: np.array, predictions2: np.array, y1: np.array = None, y2: np.array = None,
            sample_weights1: np.array = None, sample_weights2: np.array = None):
        '''
        :param predictions1: if task == 'classification'
        if prediction_type == 'label', y_pred should be with labels in (0, ..., n_class - 1). If y_pred
        is predicted proba (or 'raw') (in case of multiclass), then labels should correspond to column index in y_pred.

        :param predictions2: see predictions1
        :param y1: in classification mode, values of y1 should be in (0, ..., n_class - 1) (OHE labels are not accepted)
        :param y2: see y1
        :param sample_weights1:
        :param sample_weights2:
        :return:
        '''

        # faire les checks ici
        # if dim == 1, should be of dim (n, ),
        # else, (n, n_class)
        #
        # rmk on parameter : on classif, we only accept pred_proba and raw predictions (not predicted label with threshold)
        #
        self.predictions1 = self._check_input_predictions(predictions1, 1, self.task, self.prediction_type)
        self.predictions2 = self._check_input_predictions(predictions2, 2, self.task, self.prediction_type)
        self.y1 = self._check_input_y(y1, 1, self.task, self.prediction_type)
        self.y2 = self._check_input_y(y2, 2, self.task, self.prediction_type)
        self.sample_weights1 = self._check_input_sample_weights(sample_weights1, 1, self.predictions1.shape[0])
        self.sample_weights2 = self._check_input_sample_weights(sample_weights2, 2, self.predictions2.shape[0])
        self._check_inputs_consistency()
        self._prediction_dim = self._get_prediction_dim(self.predictions1, self.predictions2, self.y1, self.y2,
                                                        task=self.task, prediction_type=self.prediction_type)
        self.class_names = self._get_class_names(self.predictions1, self.predictions2, self.y1, self.y2, self.task,
                                                 self.prediction_type, self._prediction_dim, self.class_names)

    @staticmethod
    def _check_input_task(task: str) -> str:
        if task not in ['regression', 'classification']:
            raise ValueError(f'Bad value for "task"')
        else:
            return task

    @staticmethod
    def _check_input_prediction_type(prediction_type: str, task: str) -> str:
        if task == 'classification':
            if prediction_type is None:
                raise ValueError(f'When task == "classification", "prediction_type" parameter should be specified')
            if prediction_type not in ['raw', 'proba', 'label']:
                raise ValueError(f'Bad value for "prediction_type"')
            else:
                return prediction_type
        else:  # task == 'regression'
            if prediction_type is not None:
                OutputDriftDetector.logger.warning('Provided value for "prediction_type" is override to "raw"'
                                                    'because task == "regression"')
            return 'raw'

    @staticmethod
    def _check_input_class_names(class_names: List[str]) -> List[str]:
        if class_names is None:
            return None
        else:
            return [str(x) for x in class_names]

    @staticmethod
    def _check_input_predictions(predictions: np.array, n: int, task: str, prediction_type: str) -> np.array:
        predictions = np.array(np.squeeze(predictions))
        if task == 'regression':
            if predictions.ndim != 1:
                raise ValueError(f'Bad shape for "predictions{n}"')
        else:  # task == 'classification':
            if prediction_type in ['raw', 'proba']:
                if predictions.ndim not in [1, 2]:
                    raise ValueError(f'Bad shape for "predictions{n}"')
                if predictions.ndim == 2 and predictions.shape[1] == 2:
                    predictions = predictions[:, 1]
            else:  # prediction_type == 'label'
                if predictions.ndim != 1:
                    raise ValueError(f'Bad shape for "predictions{n}"')
                predictions.astype(int, casting='safe', copy=False)
        return predictions

    @staticmethod
    def _check_input_y(y: np.array, n: int, task: str, prediction_type: str) -> np.array:
        if y is None:
            return None
        y = np.squeeze(y)
        if task == 'regression':
            if y.ndim != 1:
                raise ValueError(f'Bad shape for "y{n}"')
        else:  # task == 'classification':
            if prediction_type in ['raw', 'proba']:
                if y.ndim != 1:
                    raise ValueError(f'Bad shape for "y{n}"')
            else:  # prediction_type == 'label'
                if y.ndim != 1:
                    raise ValueError(f'Bad shape for "y{n}"')
                y.astype(int, casting='safe', copy=False)
        return y

    @staticmethod
    def _check_input_sample_weights(sample_weights: np.array, n:int, expected_length: int) -> np.array:
        if sample_weights is None:
            return np.ones(expected_length)
        sample_weights = np.squeeze(sample_weights)
        if sample_weights.ndim != 1:
            raise ValueError(f'Bad shape for "sample_weights{n}"')
        elif sample_weights.shape[0] != expected_length:
            raise ValueError(f'"sample_weights{n}" and "predictions{n}" shapes are not consistent')
        elif np.any(sample_weights < 0):
            raise ValueError(f'Elements in "sample_weights{n}" must be non negative')
        elif np.sum(sample_weights) == 0:
            raise ValueError(f'The sum of "sample_weights{n}" must be positive')
        else:
            return sample_weights * len(sample_weights) / np.sum(sample_weights)

    def _check_inputs_consistency(self):
        # check consistency between y1 == None and y2 == None
        if (self.y1 is None) != (self.y2 is None):  # xor operation
            raise ValueError(f'y1 and y2 should be either both None or non None')

        # check consistency between predictions1 and predictions2 dimensions
        if self.predictions1.ndim == 2 and self.predictions2.ndim == 2:
            if self.predictions1.shape[1] != self.predictions2.shape[1]:
                raise ValueError(f'Not the same shape for "predictions1" and "predictions2"')

        # check consistency between class_names and prediction dimensions
        if self.class_names is not None:
            # case of binary classif
            if self.predictions1.ndim == 1:
                if len(self.class_names) != 2:
                    raise ValueError(f'"predictions" inputs indicates a binary classification but "class_names"'
                                     f'do not have 2 elements')
            # case of multiclass classif
            elif len(self.class_names) != self.predictions1.shape[1]:
                raise ValueError(f'Number of elements in "class_names" do not match with number of dimensions in'
                                 f'"predictions"')

    @staticmethod
    def _get_prediction_dim(predictions1: np.array, predictions2: np.array, y1: np.array, y2: np.array,
                            task: str, prediction_type: str) -> int:
        if task == 'regression':
            return 1
        else:  # task == 'classification'
            if prediction_type in ['raw', 'proba']:
                if predictions1.ndim == 1:  # binary classif
                    return 1
                else:  # multiclass classif
                    return predictions1.shape[1]
            else:  # prediction_type == 'label'
                return len(find_uniques(predictions1, predictions2, y1, y2))

    @staticmethod
    def _get_class_names(predictions1: np.array, predictions2: np.array, y1: np.array, y2: np.array, task: str,
                         prediction_type: str, prediction_dim: int, class_names: List[str]) -> List[str]:
        if task == 'regression':
            return None
        else:  # task == 'classification
            if prediction_type in ['raw', 'proba']:
                if class_names is not None:
                    return class_names
                elif y1 is not None:
                    return [str(x) for x in sorted(find_uniques(y1, y2))]
                else:  # prediction_type == 'label'
                    n_class = 2 if prediction_dim == 1 else prediction_dim
                    return [str(i) for i in range(n_class)]
            else:
                if class_names is not None:
                    return class_names
                else:
                    return [str(x) for x in sorted(find_uniques(predictions1, predictions2, y1, y2))]

    def get_prediction_drift(self) -> List[AbstractDriftMetrics]:
        """
        Compute drift measures based on predictions.

        See the documentation in README for explanations about how it is computed,
        especially the slide presentation.

        Returns
        -------
        prediction_drift : list of dict
            Drift measures for each predicted dimension.
        """
        self._raise_no_fit_error()
        return self._compute_prediction_drift(self.predictions1, self.predictions2, self.prediction_type,
                                              self._prediction_dim, self.sample_weights1, self.sample_weights2)

    @staticmethod
    def _compute_prediction_drift(predictions1, predictions2, prediction_type, prediction_dim, sample_weights1, sample_weights2):
        prediction_drift = []
        if prediction_type in ['raw', 'proba']:
            if prediction_dim == 1:
                prediction_drift.append(compute_drift_num(predictions1, predictions2, sample_weights1, sample_weights2))
            else:  # multiclass classif
                for i in range(prediction_dim):
                    drift = compute_drift_num(predictions1[:, i], predictions2[:, i],
                                              sample_weights1, sample_weights2)
                    prediction_drift.append(drift)
        else:  # prediction_type == "label"
            prediction_drift.append(compute_drift_cat(predictions1, predictions2, sample_weights1, sample_weights2))
        return prediction_drift

    def plot_prediction_drift(self, bins: int = 10, figsize: Tuple[int, int] = (7, 5),
                              max_n_cat: int = 20,
                              legend_labels: Tuple[str, str] = ('Dataset 1', 'Dataset 2')) -> None:
        """
        Plot histogram of distribution of predictions1 and predictions2
        in order to visualize a potential drift of the predicted values.
        See the documentation in README for explanations about how it is computed,
        especially the slide presentation.

        Parameters
        ----------
        bins : int (default=100)
            "bins" parameter passed to matplotlib.pyplot.hist function.

        figsize : Tuple[int, int], optional (default=(7, 5))
            Graphic size passed to matplotlib

        max_n_cat : int (default=20)
            For multiclass classification only. Maximum number of classes to
            represent on the plot.

        legend_labels : Tuple[str, str] (default=('Dataset 1', 'Dataset 2'))
            Legend labels used for dataset 1 and dataset 2

        Returns
        -------
        None
        """
        self._raise_no_fit_error()
        if self.prediction_type in ['raw', 'proba']:
            if self._prediction_dim == 1:  # regression or binary classif
                plot_drift_num(self.predictions1, self.predictions2, self.sample_weights1, self.sample_weights2,
                               title=f'Predictions', figsize=figsize, bins=bins, legend_labels=legend_labels)
            else:  # multiclass classif
                for i in range(self._prediction_dim):
                    plot_drift_num(self.predictions1[:, i], self.predictions2[:, i], self.sample_weights1,
                                   self.sample_weights2, title=f'{self.class_names[i]}', figsize=figsize, bins=bins,
                                   legend_labels=legend_labels)
        if self.prediction_type == 'label':
            plot_drift_cat(self.predictions1, self.predictions2, self.sample_weights1, self.sample_weights2,
                           title=f'Predictions', figsize=figsize, max_n_cat=max_n_cat, legend_labels=legend_labels)

    def _raise_no_fit_error(self):
        if self.predictions1 is None or self.predictions2 is None:
            raise ValueError('You have to call the fit method first')

    def get_target_drift(self) -> AbstractDriftMetrics:
        """
        Compute drift measures for the labels y.

        For classification :
        - Wasserstein distance
        - Result of Chi2 2 sample test

        For regression:
        - Difference of means
        - Wasserstein distance
        - Result of Kolmogorov 2 sample test

        See the documentation in README for explanations about how it is computed,
        especially the slide presentation.

        Returns
        -------
        target_drift : dict
            Dictionary of drift measures for the labels
        """
        if self.y1 is None:  # then y2 is also None
            self._raise_no_target_error()
        else:
            if self.task == 'classification':
                return compute_drift_cat(self.y1, self.y2, self.sample_weights1, self.sample_weights2)
            elif self.task in ['regression', 'ranking']:
                return compute_drift_num(self.y1, self.y2, self.sample_weights1, self.sample_weights2)

    @staticmethod
    def _raise_no_target_error():
        # TODO: put this in AbstractDriftAnalyzer
        raise ValueError('Either y1 or y2 was not passed to "fit"')

    def plot_target_drift(self, max_n_cat: int = 20, figsize: Tuple[int, int] = (7, 5), bins: int = 10,
                          legend_labels: Tuple[str, str] = ('Dataset 1', 'Dataset 2')) -> None:
        # TODO: put this in AbstractDriftAnalyzer
        """
        Plot distributions of labels in order to
        visualize a potential drift of the target labels.

        Parameters
        ----------
        max_n_cat : int (default=20)
            For multiclass classification only. Maximum number of classes to
            represent on the plot.

        bins : int (default=100)
            For regression only. "bins" parameter passed to matplotlib.pyplot.hist function.

        figsize : Tuple[int, int] (default=(7, 5))
            Graphic size passed to matplotlib

        legend_labels : Tuple[str, str] (default=('Dataset 1', 'Dataset 2'))
            Legend labels used for dataset 1 and dataset 2

        Returns
        -------
        None
        """
        if self.y1 is None:  # then y2 is also None
            self._raise_no_target_error()
        if self.task == 'classification':
            plot_drift_cat(self.y1, self.y2, self.sample_weights1, self.sample_weights2, title='target',
                           max_n_cat=max_n_cat, figsize=figsize, legend_labels=legend_labels)
        elif self.task == 'regression':
            plot_drift_num(self.y1, self.y2, self.sample_weights1, self.sample_weights2, title='target',
                           figsize=figsize, bins=bins, legend_labels=legend_labels)

    def get_performance_metrics_drift(self) -> PerformanceMetricsDrift:
        """
        Compute performance metrics on dataset 1 and dataset 2.

        Returns
        -------
        Dictionary of performance metrics
        """
        if self.y1 is None or self.y2 is None:
            self._raise_no_target_error()
        if self.task == 'regression':
            return PerformanceMetricsDrift(dataset1=compute_regression_metrics(self.y1, self.predictions1,
                                                                               self.sample_weights1),
                                           dataset2=compute_regression_metrics(self.y2, self.predictions2,
                                                                               self.sample_weights2))
        else:  # task == 'classification':
            return PerformanceMetricsDrift(dataset1=compute_classification_metrics(self.y1, self.predictions1,
                                                                                   self.sample_weights1, self.class_names,
                                                                                   self.prediction_type),
                                           dataset2=compute_classification_metrics(self.y2, self.predictions2,
                                                                                   self.sample_weights2, self.class_names,
                                                                                   self.prediction_type))
