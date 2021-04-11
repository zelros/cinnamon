import numpy as np
from .utils import assert_import, record_import_error, safe_isinstance
import warnings

warnings.formatwarning = lambda msg, *args, **kwargs: str(msg) + '\n' # ignore everything except the message

# pylint: disable=unsubscriptable-object

try:
    from .. import _cext
except ImportError as e:
    record_import_error("cext", "C extension was not built during install!", e)

try:
    import pyspark
except ImportError as e:
    record_import_error("pyspark", "PySpark could not be imported!", e)

output_transform_codes = {
    "identity": 0,
    "logistic": 1,
    "logistic_nlogloss": 2,
    "squared_loss": 3
}

feature_perturbation_codes = {
    "interventional": 0,
    "tree_path_dependent": 1,
    "global_path_dependent": 2
}

class SingleTree:
    """ A single decision tree.

    The primary point of this object is to parse many different tree types into a common format.
    """
    def __init__(self, tree, normalize=False, scaling=1.0, data=None, data_missing=None):
        assert_import("cext")

        if safe_isinstance(tree, ["sklearn.tree._tree.Tree", "econml.tree._tree.Tree"]):
            self.children_left = tree.children_left.astype(np.int32)
            self.children_right = tree.children_right.astype(np.int32)
            self.children_default = self.children_left # missing values not supported in sklearn
            self.features = tree.feature.astype(np.int32)
            self.thresholds = tree.threshold.astype(np.float64)
            self.values = tree.value.reshape(tree.value.shape[0], tree.value.shape[1] * tree.value.shape[2])
            if normalize:
                self.values = (self.values.T / self.values.sum(1)).T
            self.values = self.values * scaling
            self.node_sample_weight = tree.weighted_n_node_samples.astype(np.float64)

        elif type(tree) is dict and 'features' in tree:
            self.children_left = tree["children_left"].astype(np.int32)
            self.children_right = tree["children_right"].astype(np.int32)
            self.children_default = tree["children_default"].astype(np.int32)
            self.features = tree["features"].astype(np.int32)
            self.thresholds = tree["thresholds"]
            self.values = tree["values"] * scaling
            self.node_sample_weight = tree["node_sample_weight"]

        # deprecated dictionary support (with sklearn singlular style "feature" and "value" names)
        elif type(tree) is dict and 'children_left' in tree:
            self.children_left = tree["children_left"].astype(np.int32)
            self.children_right = tree["children_right"].astype(np.int32)
            self.children_default = tree["children_default"].astype(np.int32)
            self.features = tree["feature"].astype(np.int32)
            self.thresholds = tree["threshold"]
            self.values = tree["value"] * scaling
            self.node_sample_weight = tree["node_sample_weight"]

        elif safe_isinstance(tree, "pyspark.ml.classification.DecisionTreeClassificationModel") \
                or safe_isinstance(tree, "pyspark.ml.regression.DecisionTreeRegressionModel"):
            #model._java_obj.numNodes() doesn't give leaves, need to recompute the size
            def getNumNodes(node, size):
                size = size + 1
                if node.subtreeDepth() == 0:
                    return size
                else:
                    size = getNumNodes(node.leftChild(), size)
                    return getNumNodes(node.rightChild(), size)

            num_nodes = getNumNodes(tree._java_obj.rootNode(), 0)
            self.children_left = np.full(num_nodes, -2, dtype=np.int32)
            self.children_right = np.full(num_nodes, -2, dtype=np.int32)
            self.children_default = np.full(num_nodes, -2, dtype=np.int32)
            self.features = np.full(num_nodes, -2, dtype=np.int32)
            self.thresholds = np.full(num_nodes, -2, dtype=np.float64)
            self.values = [-2]*num_nodes
            self.node_sample_weight = np.full(num_nodes, -2, dtype=np.float64)
            def buildTree(index, node):
                index = index + 1
                if tree._java_obj.getImpurity() == 'variance':
                    self.values[index] = [node.prediction()] #prediction for the node
                else:
                    self.values[index] = [e for e in node.impurityStats().stats()] #for gini: NDarray(numLabel): 1 per label: number of item for each label which went through this node
                self.node_sample_weight[index] = node.impurityStats().count() #weighted count of element trough this node

                if node.subtreeDepth() == 0:
                    return index
                else:
                    self.features[index] = node.split().featureIndex() #index of the feature we split on, not available for leaf, int
                    if str(node.split().getClass()).endswith('tree.CategoricalSplit'):
                        #Categorical split isn't implemented, TODO: could fake it by creating a fake node to split on the exact value?
                        raise NotImplementedError('CategoricalSplit are not yet implemented')
                    self.thresholds[index] = node.split().threshold() #threshold for the feature, not available for leaf, float

                    self.children_left[index] = index + 1
                    idx = buildTree(index, node.leftChild())
                    self.children_right[index] = idx + 1
                    idx = buildTree(idx, node.rightChild())
                    return idx

            buildTree(-1, tree._java_obj.rootNode())
            #default Not supported with mlib? (TODO)
            self.children_default = self.children_left
            self.values = np.asarray(self.values)
            if normalize:
                self.values = (self.values.T / self.values.sum(1)).T
            self.values = self.values * scaling

        elif type(tree) == dict and 'tree_structure' in tree: # LightGBM model dump
            start = tree['tree_structure']
            num_parents = tree['num_leaves']-1
            self.children_left = np.empty((2*num_parents+1), dtype=np.int32)
            self.children_right = np.empty((2*num_parents+1), dtype=np.int32)
            self.children_default = np.empty((2*num_parents+1), dtype=np.int32)
            self.features = np.empty((2*num_parents+1), dtype=np.int32)
            self.thresholds = np.empty((2*num_parents+1), dtype=np.float64)
            self.values = [-2]*(2*num_parents+1)
            self.node_sample_weight = np.empty((2*num_parents+1), dtype=np.float64)
            visited, queue = [], [start]
            while queue:
                vertex = queue.pop(0)
                if 'split_index' in vertex.keys():
                    if vertex['split_index'] not in visited:
                        if 'split_index' in vertex['left_child'].keys():
                            self.children_left[vertex['split_index']] = vertex['left_child']['split_index']
                        else:
                            self.children_left[vertex['split_index']] = vertex['left_child']['leaf_index']+num_parents
                        if 'split_index' in vertex['right_child'].keys():
                            self.children_right[vertex['split_index']] = vertex['right_child']['split_index']
                        else:
                            self.children_right[vertex['split_index']] = vertex['right_child']['leaf_index']+num_parents
                        if vertex['default_left']:
                            self.children_default[vertex['split_index']] = self.children_left[vertex['split_index']]
                        else:
                            self.children_default[vertex['split_index']] = self.children_right[vertex['split_index']]
                        self.features[vertex['split_index']] = vertex['split_feature']
                        self.thresholds[vertex['split_index']] = vertex['threshold']
                        self.values[vertex['split_index']] = [vertex['internal_value']]
                        self.node_sample_weight[vertex['split_index']] = vertex['internal_count']
                        visited.append(vertex['split_index'])
                        queue.append(vertex['left_child'])
                        queue.append(vertex['right_child'])
                else:
                    self.children_left[vertex['leaf_index']+num_parents] = -1
                    self.children_right[vertex['leaf_index']+num_parents] = -1
                    self.children_default[vertex['leaf_index']+num_parents] = -1
                    self.features[vertex['leaf_index']+num_parents] = -1
                    self.children_left[vertex['leaf_index']+num_parents] = -1
                    self.children_right[vertex['leaf_index']+num_parents] = -1
                    self.children_default[vertex['leaf_index']+num_parents] = -1
                    self.features[vertex['leaf_index']+num_parents] = -1
                    self.thresholds[vertex['leaf_index']+num_parents] = -1
                    self.values[vertex['leaf_index']+num_parents] = [vertex['leaf_value']]
                    self.node_sample_weight[vertex['leaf_index']+num_parents] = vertex['leaf_count']
            self.values = np.asarray(self.values)
            self.values = np.multiply(self.values, scaling)

        elif type(tree) == dict and 'nodeid' in tree:
            """ Directly create tree given the JSON dump (with stats) of a XGBoost model.
            """

            def max_id(node):
                if "children" in node:
                    return max(node["nodeid"], *[max_id(n) for n in node["children"]])
                else:
                    return node["nodeid"]

            m = max_id(tree) + 1
            self.children_left = -np.ones(m, dtype=np.int32)
            self.children_right = -np.ones(m, dtype=np.int32)
            self.children_default = -np.ones(m, dtype=np.int32)
            self.features = -np.ones(m, dtype=np.int32)
            self.thresholds = np.zeros(m, dtype=np.float64)
            self.values = np.zeros((m, 1), dtype=np.float64)
            self.node_sample_weight = np.empty(m, dtype=np.float64)

            def extract_data(node, tree):
                i = node["nodeid"]
                tree.node_sample_weight[i] = node["cover"]

                if "children" in node:
                    tree.children_left[i] = node["yes"]
                    tree.children_right[i] = node["no"]
                    tree.children_default[i] = node["missing"]
                    tree.features[i] = node["split"]
                    tree.thresholds[i] = node["split_condition"]

                    for n in node["children"]:
                        extract_data(n, tree)
                elif "leaf" in node:
                    tree.values[i] = node["leaf"] * scaling

            extract_data(tree, self)

        elif type(tree) == str:
            """ Build a tree from a text dump (with stats) of xgboost.
            """

            nodes = [t.lstrip() for t in tree[:-1].split("\n")]
            nodes_dict = {}
            for n in nodes: nodes_dict[int(n.split(":")[0])] = n.split(":")[1]
            m = max(nodes_dict.keys())+1
            children_left = -1*np.ones(m,dtype="int32")
            children_right = -1*np.ones(m,dtype="int32")
            children_default = -1*np.ones(m,dtype="int32")
            features = -2*np.ones(m,dtype="int32")
            thresholds = -1*np.ones(m,dtype="float64")
            values = 1*np.ones(m,dtype="float64")
            node_sample_weight = np.zeros(m,dtype="float64")
            values_lst = list(nodes_dict.values())
            keys_lst = list(nodes_dict.keys())
            for i in range(0,len(keys_lst)):
                value = values_lst[i]
                key = keys_lst[i]
                if ("leaf" in value):
                    # Extract values
                    val = float(value.split("leaf=")[1].split(",")[0])
                    node_sample_weight_val = float(value.split("cover=")[1])
                    # Append to lists
                    values[key] = val
                    node_sample_weight[key] = node_sample_weight_val
                else:
                    c_left = int(value.split("yes=")[1].split(",")[0])
                    c_right = int(value.split("no=")[1].split(",")[0])
                    c_default = int(value.split("missing=")[1].split(",")[0])
                    feat_thres = value.split(" ")[0]
                    if ("<" in feat_thres):
                        feature = int(feat_thres.split("<")[0][2:])
                        threshold = float(feat_thres.split("<")[1][:-1])
                    if ("=" in feat_thres):
                        feature = int(feat_thres.split("=")[0][2:])
                        threshold = float(feat_thres.split("=")[1][:-1])
                    node_sample_weight_val = float(value.split("cover=")[1].split(",")[0])
                    children_left[key] = c_left
                    children_right[key] = c_right
                    children_default[key] = c_default
                    features[key] = feature
                    thresholds[key] = threshold
                    node_sample_weight[key] = node_sample_weight_val

            self.children_left = children_left
            self.children_right = children_right
            self.children_default = children_default
            self.features = features
            self.thresholds = thresholds
            self.values = values[:,np.newaxis] * scaling
            self.node_sample_weight = node_sample_weight
        else:
            raise Exception("Unknown input to SingleTree constructor: " + str(tree))

        # Re-compute the number of samples that pass through each node if we are given data
        if data is not None and data_missing is not None:
            self.node_sample_weight[:] = 0.0
            _cext.dense_tree_update_weights(
                self.children_left, self.children_right, self.children_default, self.features,
                self.thresholds, self.values, 1, self.node_sample_weight, data, data_missing
            )

        # we compute the expectations to make sure they follow the SHAP logic
        self.max_depth = _cext.compute_expectations(
            self.children_left, self.children_right, self.node_sample_weight,
            self.values
        )

class IsoTree(SingleTree):
    """
    In sklearn the tree of the Isolation Forest does not calculated in a good way.
    """
    def __init__(self, tree, tree_features, normalize=False, scaling=1.0, data=None, data_missing=None):
        super(IsoTree, self).__init__(tree, normalize, scaling, data, data_missing)
        if safe_isinstance(tree, "sklearn.tree._tree.Tree"):
            from sklearn.ensemble._iforest import _average_path_length # pylint: disable=no-name-in-module

            def _recalculate_value(tree, i , level):
                if tree.children_left[i] == -1 and tree.children_right[i] == -1:
                    value = level + _average_path_length(np.array([tree.n_node_samples[i]]))[0]
                    self.values[i, 0] =  value
                    return value * tree.n_node_samples[i]
                else:
                    value_left = _recalculate_value(tree, tree.children_left[i] , level + 1)
                    value_right = _recalculate_value(tree, tree.children_right[i] , level + 1)
                    self.values[i, 0] =  (value_left + value_right) / tree.n_node_samples[i]
                    return value_left + value_right

            _recalculate_value(tree, 0, 0)
            if normalize:
                self.values = (self.values.T / self.values.sum(1)).T
            self.values = self.values * scaling
            # re-number the features if each tree gets a different set of features
            self.features = np.where(self.features >= 0, tree_features[self.features], self.features)
