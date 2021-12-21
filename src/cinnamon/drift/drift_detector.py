
class DriftDetector:

    def __init__(self, ):
        self.predictions1 = None
        self.predictions2 = None
        self.y1 = None
        self.y2 = None
        self.sample_weights1 = None
        self.sample_weights2 = None

    def fit(self, predictions1: np.array, predictions2: np.array, y1: np.array = None, y2: np.array = None,
            sample_weights1: np.array = None, sample_weights2: np.array = None):

        # faire les checks ici
        # if dim == 1, should be of dim (n, ),
        # else, (n, n_class)

        self.prediction_dim = ???
        self.predictions1 = predictions1
        self.predictions2 = predictions2
        self.y1 = y1
        self.y2 = y2
        self.sample_weights1 = sample_weights1
        self.sample_weights2 = sample_weights2


    def get_prediction_drift(self, prediction_type: str = "raw") -> List[dict]:
        # TODO: come from ModelDriftExplainer
        """
        Compute drift measures based on model predictions.

        See the documentation in README for explanations about how it is computed,
        especially the slide presentation.

        Returns
        -------
        prediction_drift : list of dict
            Drift measures for each predicted dimension.
        """
        if prediction_type not in ['raw', 'proba']:
            raise ValueError(f'Bad value for prediction_type: {prediction_type}')
        if prediction_type == 'raw':
            return self._compute_prediction_drift(self.predictions1, self.predictions2, self.task, self._prediction_dim,
                                                  self.sample_weights1, self.sample_weights2)
        elif prediction_type == 'proba':
            return self._compute_prediction_drift(self.pred_proba1, self.pred_proba2, self.task, self._prediction_dim,
                                                  self.sample_weights1, self.sample_weights2)

    @staticmethod
    def _compute_prediction_drift(predictions1, predictions2, task, prediction_dim, sample_weights1=None, sample_weights2=None):
        # TODO: come from ModelDriftExplainer
        prediction_drift = []
        if task == 'classification':
            if prediction_dim == 1:  # binary classif
                prediction_drift.append(compute_drift_num(predictions1, predictions2, sample_weights1, sample_weights2))
            else:  # multiclass classif
                for i in range(predictions1.shape[1]):
                    drift = compute_drift_num(predictions1[:, i], predictions2[:, i],
                                              sample_weights1, sample_weights2)
                    prediction_drift.append(drift)
        elif task in ['regression', 'ranking']:
            prediction_drift.append(compute_drift_num(predictions1, predictions2, sample_weights1, sample_weights2))
        return prediction_drift

    def plot_prediction_drift(self, prediction_type='raw', bins: int = 10,
                              figsize: Tuple[int, int] = (7, 5)) -> None:
        # TODO: come from ModelDriftExplainer
        """
        Plot histogram of distribution of predictions for dataset 1 and dataset 2
        in order to visualize a potential drift of the predicted values.
        See the documentation in README for explanations about how it is computed,
        especially the slide presentation.

        Parameters
        ----------
        prediction_type: str, optional (default="raw")
            Type of predictions to consider.
            Choose among:
            - "raw" : logit predictions (binary classification), log-softmax predictions
            (multiclass classification), regular predictions (regression)
            - "proba" : predicted probabilities (only for classification models)

        bins : int (default=100)
            "bins" parameter passed to matplotlib.pyplot.hist function.

        figsize : Tuple[int, int], optional (default=(7, 5))
            Graphic size passed to matplotlib

        Returns
        -------
        None
        """
        if self.predictions1 is None:
            raise ValueError('You must call the fit method before ploting drift')
        if prediction_type not in ['raw', 'proba']:
            raise ValueError(f'Bad value for prediction_type: {prediction_type}')
        if prediction_type == 'raw':
            pred1, pred2 = self.predictions1, self.predictions2
        else:
            pred1, pred2 = self.pred_proba1, self.pred_proba2

        if self.task == 'classification' and self._model_parser.prediction_dim > 1:  # multiclass classif
            for i in range(self._model_parser.prediction_dim):
                plot_drift_num(pred1[:, i], pred2[:, i], self.sample_weights1, self.sample_weights2,
                               title=f'{self.class_names[i]}', figsize=figsize, bins=bins)
        else:  # binary classif or regression
            plot_drift_num(pred1, pred2, self.sample_weights1, self.sample_weights2, title=f'Predictions',
                           figsize=figsize, bins=bins)

    def get_target_drift(self) -> dict:
        # TODO: put this in AbstractDriftAnalyzer
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
        if isinstance(self.target_drift, dict):
            return self.target_drift

        if self.y1 is None or self.y2 is None:
            self._raise_no_target_error()
        else:
            if self.task == 'classification':
                self.target_drift = compute_drift_cat(self.y1, self.y2, self.sample_weights1, self.sample_weights2)
            elif self.task in ['regression', 'ranking']:
                self.target_drift = compute_drift_num(self.y1, self.y2, self.sample_weights1, self.sample_weights2)
        return self.target_drift

    @staticmethod
    def _raise_no_target_error():
        # TODO: put this in AbstractDriftAnalyzer
        raise ValueError('Either y1 or y2 was not passed in DriftExplainer.fit')

    def plot_target_drift(self, max_n_cat: int = 20, figsize: Tuple[int, int] = (7, 5), bins: int = 10):
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

        Returns
        -------
        None
        """
        if self.y1 is None or self.y2 is None:
            raise ValueError('"y1" or "y2" argument was not passed to drift_explainer.fit method')
        if self.task == 'classification':
            plot_drift_cat(self.y1, self.y2, self.sample_weights1, self.sample_weights2, title='target',
                           max_n_cat=max_n_cat, figsize=figsize)
        elif self.task == 'regression':
            plot_drift_num(self.y1, self.y2, self.sample_weights1, self.sample_weights2, title='target',
                           figsize=figsize, bins=bins)

    @staticmethod
    def _check_sample_weights(sample_weights, X):
        # TODO: put this in AbstractDriftAnalyzer
        if sample_weights is None:
            return np.ones(X.shape[0])
        elif np.any(sample_weights < 0):
            raise ValueError("Elements in sample_weights must be non negative")
        elif np.sum(sample_weights) == 0:
            raise ValueError("The sum of sample_weights must be positive")
        else:
            return sample_weights * len(sample_weights) / np.sum(sample_weights)
