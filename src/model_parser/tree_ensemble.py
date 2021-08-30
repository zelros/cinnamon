from abc import ABC


class TreeEnsembleParser(ABC):
    def __init__(self):
        self.num_trees = None
        self.max_depth = None
        self.feature_names = None
        self.cat_features = None
        self.n_features = None
        self.trees = None
        self.class_names = None
        self.model_objective = None
