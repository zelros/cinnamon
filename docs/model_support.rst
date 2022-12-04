==========================
Supported Models
==========================

Many CinnaMon features are model agnostic: estimate drift of inputs, target, predictions of the model/pipeline.

But for drift explainability CinnaMon provides both model agnostic methods
(which use the model/pipeline as a black box) and model specific methods 
(which leverage the model structure).

Model Specific Support
=================================================

Currenlty, model specific method only exists for tree based models and tree ensemble
models. Implementation is available for **XGBoost** and **CatBoost**.

Tree based computation of drift importances is available under the 
``get_tree_based_drift_importances`` method.

An explanation of how tree based drift importances are computed is given in the 
`slide presentation <https://yohannlefaou.github.io/publications/2021-cinnamon/Detect_explain_and_correct_data_drift_in_a_machine_learning_system.pdf>`_.


Model Agnostic Support
==================================================

Model agnostic computation of drift importances is available under the 
``get_model_agnostic_drift_importances`` method.

This mode can handle both machine learning models and machine learning pipeline 
as soon as the ``model`` object passed as argument to ``ModelDriftExplainer`` has the following
method:

- if ``task == "regression"``, ``model`` should implement the ``predict`` method which returns
  the raw predictions of the model/pipeline.
- if ``task == "classification"``, ``model`` should implement the following methods:
  
  - ``predict_proba``: should return the predicted probabilities of the model/pipeline.
  - ``predict_raw``: Optional, should return the logit (binary classification) or log-softmax
    (multiclass classification) predictions of the model/pipeline.
  - ``predict``: Optional, should return the predicted class of the model/pipeline.

  If not provided, ``predict_raw`` is inferred from ``predict_proba``.


As a result, user can wrap any model/pipeline into a class which implements 
the above methods, and pass the wraper as input to ``ModelDriftExplainer``. 
An example of ML pipeline used with CinnaMon is given in `this notebook <https://github.com/zelros/cinnamon/blob/ac729da6a00ef07dda37f912a6e1297cb68e184d/examples/amesHousing_LinearRegression_ModelDriftExplainer.ipynb>`_.
