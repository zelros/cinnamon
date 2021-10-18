from .i_model_parser import IModelParser


class ITreeEnsembleParser(IModelParser):
    def __init__(self):
        super().__init__()
        self.n_trees = None
        self.max_depth = None
        self.trees = None
        self.model_objective = None
        self.n_iterations = None
        self.base_score = None  # bias
        self.node_weights1 = None
        self.node_weights2 = None
