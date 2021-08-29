import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from pweave import weave
from os import path
import pkgutil

from scipy.stats import wasserstein_distance, ks_2samp
from .tree_ensemble import CatBoostParser
from .utils import wasserstein_distance_for_cat, chi2_test, compute_distribution_cat


logging.basicConfig(format='%(levelname)s:%(asctime)s - (%(pathname)s) %(message)s', level=logging.INFO)


def compute_drift_num(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None):
    if sample_weights1 is None and sample_weights2 is None: # TODO: does ks generalize to weighted samples ?
        kolmogorov_smirnov = ks_2samp(a1, a2)
    else:
        kolmogorov_smirnov = None
    return {'wasserstein': wasserstein_distance(a1, a2, sample_weights1, sample_weights2),
            'kolmogorov_smirnov': kolmogorov_smirnov}


def compute_drift_cat(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None):
    if sample_weights1 is None and sample_weights2 is None:
        # TODO: does chi2 generalize to weighted samples ?
        # indeed, I am sure it is not sufficient to compute the contingency table with weights. So the chi2 formula need
        # to take weights into account
        # chi2 should not take min_cat_weight into account. If pbm with number of cat, should be handled by
        # chi2_test internally with a proper solution

        # TODO chi2 not working for now
        #chi2 = chi2_test(np.concatenate((a1, a2)), np.array([0] * len(a1) + [1] * len(a2)))
        chi2 = None
    else:
        chi2 = None
    return {'wasserstein': wasserstein_distance_for_cat(a1, a2, sample_weights1, sample_weights2),
            'chi2_test': chi2}


def plot_drift_cat(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None, title=None,
                   min_cat_weight: float = None):

    # compute both distributions
    distrib = compute_distribution_cat(a1, a2, sample_weights1, sample_weights2, min_cat_weight)
    bar_height = np.array([v for v in distrib.values()]) # len(distrib) rows and 2 columns

    #plot
    index = np.arange(len(distrib))
    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(index, bar_height[:, 0], bar_width, label="Dataset 1")
    ax.bar(index+bar_width, bar_height[:, 1], bar_width, label="Dataset 2")

    ax.set_xlabel('Category')
    ax.set_ylabel('Percentage')
    ax.set_title(title)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(list(distrib.keys()), rotation=30)
    ax.legend()
    plt.show()


def plot_drift_num(a1: np.array, a2: np.array, sample_weights1: np.array=None, sample_weights2: np.array=None,
                   title=None):
    #distrib = compute_distribution_num(a1, a2, sample_weights1, sample_weights2)
    plt.hist(a1, bins=100, density=True, weights=sample_weights1, alpha=0.3)
    plt.hist(a2, bins=100, density=True, weights=sample_weights2, alpha=0.3)
    plt.legend(['Dataset 1', 'Dataset 2'])
    plt.title(title)
    plt.show()


def compute_distribution_num(a1: np.array, a2: np.array, sample_weights1=None, sample_weights2=None):
    pass


class DriftExplainer:

    logger = logging.getLogger('DriftExplainer')

    def __init__(self):
        #self.parsed_model = None
        #self.node_weights1 = None
        #self.node_weights2 = None
        self.feature_drifts = None
        self.target_drift = None
        self.prediction_drift = None
        self.feature_contribs = None
        self.feature_names = None
        self.cat_features = None
        self.num_features = None
        self.predictions1 = None
        self.predictions2 = None
        self.model_objective = None
        #self.cat_feature_distribs = None

        # usefull for the report when I export the dataset (may not be usefull if I stock only the necessary info
        # to make the plot
        self.X1 = None # same
        self.X2 = None # same
        self.sample_weights1 = None # same
        self.sample_weights2 = None # same
        self.y1 = None # same
        self.y2 = None # same

    def fit(self, model, X1: pd.DataFrame, X2: pd.DataFrame, y1: np.array=None, y2: np.array= None,
            sample_weights1: np.array = None, sample_weights2: np.array = None):
        """
        Compute drift coefficients by feature. For algebraic drift, we do X2 - X1
        :param model: the (tree based) model we want to analyze the drift on
        :param X1: the base X in the comparison (usually the train X or validation X)
        :param X2: the X which is compared with X1 (usually the test X of production X)
        :return:
        """
        if (sample_weights1 is not None and np.any(sample_weights1 < 0)) or (sample_weights2 is not None and
                                                                             np.any(sample_weights2 < 0)):
            raise ValueError("Elements in sample_weights must be non negative")
        if (sample_weights1 is not None and np.sum(sample_weights1) == 0) or (sample_weights2 is not None and
                                                                              np.sum(sample_weights2) == 0):
            raise ValueError("The sum of sample_weights must be positive")

        DriftExplainer.logger.info('Step 1 - Parse model')
        self.parsed_model = self._parse_model(model)
        self.feature_names = self.parsed_model.feature_names
        self.cat_features = self.parsed_model.cat_features
        self.num_features = [f for f in self.feature_names if f not in self.cat_features]
        self.model_objective = self.parsed_model.model_objective

        # Compute predictions
        self.predictions1 = self.parsed_model.get_predictions(X1[self.cat_features + self.num_features])
        self.predictions2 = self.parsed_model.get_predictions(X2[self.cat_features + self.num_features])

        # Drift of the distribution of predictions
        self.prediction_drift = self._compute_prediction_drift(self.predictions1, self.predictions2,
                                                               self.model_objective, self.parsed_model.class_names,
                                                               sample_weights1, sample_weights2)

        # Drift of each feature of the model
        self.feature_drifts = self._compute_feature_drifts(X1, X2, self.feature_names, self.cat_features,
                                                           sample_weights1, sample_weights2)

        # Drift of the target ground truth labels
        self.target_drift = self._compute_target_drift(y1, y2, self.model_objective, sample_weights1, sample_weights2)

        # Compute node weights: weighted sum of observations in each node
        DriftExplainer.logger.info('Step 2 - Pass datasets through the model trees')
        self.node_weights1 = self.parsed_model.get_node_weights(X1[self.cat_features + self.num_features],
                                                                sample_weights=sample_weights1)
        self.node_weights2 = self.parsed_model.get_node_weights(X2[self.cat_features + self.num_features],
                                                                sample_weights=sample_weights2)

        # Explain prediction drift by decompose it in feature contributions
        DriftExplainer.logger.info('Compute feature contributions to the drift')
        #split_features_index_list = np.array([])
        self.feature_contribs = np.zeros((self.parsed_model.n_features, len(self.parsed_model.class_names)))
        for i, tree in enumerate(self.parsed_model.trees):
            #split_features_index_list = np.concatenate((split_features_index_list, tree.split_features_index), axis=0)
            feature_contribs_tree = tree.compute_feature_contribs(self.node_weights1[i],
                                                                  self.node_weights2[i],
                                                                  n_features=self.parsed_model.n_features)
            self.feature_contribs += feature_contribs_tree

        # temporary
        self.X1 = X1
        self.X2 = X2
        self.sample_weights1 = sample_weights1
        self.sample_weights2 = sample_weights2
        self.y1 = y1
        self.y2 = y2

    @staticmethod
    def _compute_prediction_drift(predictions1, predictions2, model_objective, class_names=None,
                                  sample_weights1=None, sample_weights2=None):
        DriftExplainer.logger.info('Evaluate prediction drift')
        prediction_drift = []
        if model_objective == 'multiclass_classification':
            for i, label in enumerate(class_names):
                label_drift = compute_drift_num(predictions1[:, i], predictions2[:, i],
                                                sample_weights1, sample_weights2)
                label_drift['label'] = label
                prediction_drift.append(label_drift)
        elif model_objective == 'binary_classification':
            pass
        elif model_objective == 'regression':
            pass
        return prediction_drift

    @staticmethod
    def _compute_feature_drifts(X1, X2, feature_names, cat_features, sample_weights1, sample_weights2):
        DriftExplainer.logger.info('Evaluate univariate drift of each feature')
        feature_drifts = []
        for feature_name in feature_names:
            if feature_name in cat_features:
                feature_drift = compute_drift_cat(X1[feature_name].values, X2[feature_name].values,
                                                  sample_weights1, sample_weights2)
            else:
                feature_drift = compute_drift_num(X1[feature_name].values, X2[feature_name].values,
                                                  sample_weights1, sample_weights2)
            feature_drifts.append(feature_drift)
        return feature_drifts

    @staticmethod
    def _compute_target_drift(y1, y2, model_objective, sample_weights1, sample_weights2):
        if y1 is not None and y2 is not None:
            DriftExplainer.logger.info('Evaluate drift of the target ground truth labels')
            if model_objective in ['binary_classification', 'multiclass_classification']:
                return compute_drift_cat(y1, y2, sample_weights1, sample_weights2)
            elif model_objective == 'regression':
                return compute_drift_num(y1, y2, sample_weights1, sample_weights2)
            else:
                return None

    def plot_target_drift(self, min_cat_weight: float = 0.01):
        if self.y1 is None or self.y2 is None:
            raise ValueError('"y1" or "y2" argument was not passed to drift_explainer.fit method')
        if self.model_objective == 'multiclass_classification':
            plot_drift_cat(self.y1, self.y2, self.sample_weights1, self.sample_weights2, title='target',
                           min_cat_weight=min_cat_weight)
        elif self.model_objective == 'binary_classification':
            pass
        elif self.model_objective == 'regression':
            pass

    def plot_prediction_drift(self):
        if self.predictions1 is None:
            raise ValueError('You must call the fit method before ploting drift')
        if self.model_objective == 'multiclass_classification':
            for i, label in enumerate(self.parsed_model.class_names):
                plt.hist(self.predictions1[:, i], bins=100, density=True, alpha=0.3)
                plt.hist(self.predictions2[:, i], bins=100, density=True, alpha=0.3)
                plt.title(f'{label}')
                plt.legend(['dataset1', 'dataset2'])
                plt.show()
        elif self.model_objective == 'binary_classification':
            pass
        elif self.model_objective == 'regression':
            pass

    def plot_feature_drift(self, feature_name, min_cat_weight: float = 0.01):
        if self.feature_names is None:
            raise ValueError('You must call the fit method before ploting drift')
        if feature_name not in self.feature_names:
            raise ValueError(f'{feature_name} not present in the feature_names list')
        elif feature_name in self.cat_features:
            plot_drift_cat(self.X1[feature_name].values, self.X2[feature_name].values, self.sample_weights1,
                           self.sample_weights2, title=feature_name, min_cat_weight=min_cat_weight)
        else:
            plot_drift_num(self.X1[feature_name].values, self.X2[feature_name].values, self.sample_weights1,
                           self.sample_weights2, title=feature_name)

    def generate_html_report(self, path, min_cat_weight: float = 0.01):
        if self.prediction_drift is None:
            raise ValueError('You must call the fit method before generating the drift report')
        with open('report_data.pkl', 'wb') as f:
            pkl.dump({'drift_explainer': self, 'min_cat_weight': min_cat_weight}, f)
        data = pkgutil.get_data(__name__, '/report/drift_report_template.pmd')
        import tempfile
        with tempfile.NamedTemporaryFile('w') as fp:
            fp.write(data.decode('utf-8'))
            weave(fp.name, informat='markdown', # on part de l'endroit où le code est exécuter donc dans le notebook
                  output=path)

        #with open('template.pmd', 'w') as f:
        #    f.write(data.decode('utf-8'))
        #print(data)
        #print(type(data))
        #print(data.decode('utf-8'))
        #print(type(data.decode('utf-8')))
        weave('template.pmd',  # on part de l'endroit où le code est exécuter donc dans le notebook
              output=path)

        #weave('src/report/drift_report_template.pmd', # on part de l'endroit où le code est exécuter donc dans le notebook
        #      output=path)

    def update(self, new_X):
        """usually new_X would update X2: the production X"""
        pass

    def plot_feature_contribs(self, n: int = 10):
        # a voir si je veux rendre cette fonction plus générique
        if self.feature_contribs is None:
            raise ValueError('You need to run drift_explainer.fit before you can plot feature_contribs')

        feature_contribs = self.feature_contribs
        # sort by importance in terms of drift
        # sort in decreasing order according to sum of absolute values of feature_contribs
        order = np.abs(feature_contribs).sum(axis=1).argsort()[::-1].tolist()
        ordered_names = [self.feature_names[i] for i in order]
        ordered_feature_contribs = feature_contribs[order, :]

        n_class = len(self.parsed_model.class_names)

        # plot
        fig, ax = plt.subplots(figsize=(10, 10))
        X = np.arange(n)
        for i in range(n_class):
            ax.barh(X + (n_class-i-1)/(n_class+1), ordered_feature_contribs[:n,i][::-1], height=1/(n_class+1))
        ax.legend(self.parsed_model.class_names)
        ax.set_yticks(X + 1/(n_class+1) * (n_class-1)/2)
        ax.set_yticklabels(ordered_names[:n][::-1])
        ax.set_xlabel('Contribution to data drift', fontsize=15)
        plt.show()

    @staticmethod
    def _parse_model(model):
        if type(model).__name__ == 'CatBoostClassifier':
            return CatBoostParser(model)
        else:
            raise TypeError(f'The type of model {type(model).__name__} is not supported by our package') # TODO : change by package name
