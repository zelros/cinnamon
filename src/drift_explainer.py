
import logging
import numpy as np
from .tree_ensemble import CatBoostParser
import matplotlib.pyplot as plt

logging.basicConfig(format='%(levelname)s:%(asctime)s - (%(pathname)s) %(message)s', level=logging.DEBUG)


class DriftExplainer:
    logger = logging.getLogger('ExplainerForTree')

    def __init__(self):
        pass

    def fit(self, model, dataset1, dataset2):
        """
        Compute drift coefficients by feature. For algebraic drift, we do dataset2 - dataset1
        :param model: the (tree based) model we want to analyze the drift on
        :param dataset1: the the base dataset in the comparison (usually the train dataset or validation dataset)
        :param dataset2: the dataset which is compared with dataset1 (usually the test dataset of production dataset)
        :return:
        """
        print(0)
        self.parsed_model = self._parse_model(model)
        self.sample_weights_dataset1 = self.parsed_model.get_sample_weights(dataset1)
        self.sample_weights_dataset2 = self.parsed_model.get_sample_weights(dataset2)
        print(1)
        self.compute_feature_drift_values()
        print(2)

    def update(self, new_dataset):
        """usually new_dataset would update dataset2: the production dataset"""
        pass

    def compute_feature_drift_values(self):
        #split_features_index_list = np.array([])
        self.feature_drift_values = np.zeros((self.parsed_model.n_features, len(self.parsed_model.class_names)))
        for i, tree in enumerate(self.parsed_model.trees):
            #split_features_index_list = np.concatenate((split_features_index_list, tree.split_features_index), axis=0)
            feature_drift_values_tree = tree.compute_feature_drift_values(self.sample_weights_dataset1[i],
                                                                          self.sample_weights_dataset2[i],
                                                                          n_features=self.parsed_model.n_features)
            self.feature_drift_values += feature_drift_values_tree

    def plot_feature_drift_values(self, n: int = 10):
        # a voir si je veux rendre cette fonction plus générique
        if self.feature_drift_values is None:
            raise ValueError('You need to run drift_explainer.fit before you can plot feature_drift_values')

        feature_drift_values = self.feature_drift_values
        # sort by importance in terms of drift
        # sort in decreasing order according to sum of absolute values of feature_drift_values
        order = np.abs(feature_drift_values).sum(axis=1).argsort()[::-1].tolist()
        ordered_names = [self.parsed_model.feature_names[i] for i in order]
        ordered_feature_drift_values = feature_drift_values[order, :]

        n_class = len(self.parsed_model.class_names)

        # plot
        fig, ax = plt.subplots(figsize=(10, 10))
        X = np.arange(n)
        for i in range(n_class):
            ax.barh(X + (n_class-i-1)/(n_class+1), ordered_feature_drift_values[:n,i][::-1], height=1/(n_class+1))
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

