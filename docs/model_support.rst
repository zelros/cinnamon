==========================
Supported Models
==========================

Model Specific Methods
=========================

CinnaMon implements model specific techniques for drift explainability and drift correction.


- Tree based method ``get_tree_based_drift_importances`` is available to compute drift importances

  - Currently only **XGBoost** and **CatBoost** are supported


Model Agnostic Methods
===========================

Model agnostic methods are also available and rely only on call to ``model.predict``

See example in notebook ...

Pipeline are also suported: see `this example notebook <https://github.com/zelros/cinnamon/blob/ac729da6a00ef07dda37f912a6e1297cb68e184d/examples/amesHousing_LinearRegression_ModelDriftExplainer.ipynb>`_

