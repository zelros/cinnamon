.. -*- mode: rst -*-

|ReadTheDocs|_ |License|_ |PyPi|_

.. |ReadTheDocs| image:: https://readthedocs.org/projects/cinnamon/badge
.. _ReadTheDocs: https://cinnamon.readthedocs.io/en/add-documentation

.. |License| image:: https://img.shields.io/badge/License-MIT-yellow
.. _License: https://github.com/zelros/cinnamon/blob/master/LICENSE.txt

.. |PyPi| image:: https://img.shields.io/pypi/v/cinnamon
.. _PyPi: https://pypi.org/project/cinnamon/


CinnaMon
============

**CinnaMon** is a Python library which allows to monitor data drift on a 
machine learning system. It provides tools to study data drift between two datasets,
especially to detect, explain, and correct data drift.

⚡️ Quickstart
===============

As a quick example, let's illustrate the use of CinnaMon on the breast cancer data where we voluntarily introduce some data drift.

Setup the data and build a model
------------------------------------

.. code:: python

    >>> import pandas as pd
    >>> from sklearn import datasets
    >>> from sklearn.model_selection import train_test_split
    >>> from xgboost import XGBClassifier

    # load breast cancer data
    >>> dataset = datasets.load_breast_cancer()
    >>> X = pd.DataFrame(dataset.data, columns = dataset.feature_names)
    >>> y = dataset.target

    # split data in train and valid dataset
    >>> X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=2021)

    # introduce some data drift in valid by filtering with 'worst symmetry' feature
    >>> y_valid = y_valid[X_valid['worst symmetry'].values > 0.3]
    >>> X_valid = X_valid.loc[X_valid['worst symmetry'].values > 0.3, :].copy()

    # fit a XGBClassifier on the training data
    >>> clf = XGBClassifier(use_label_encoder=False)
    >>> clf.fit(X=X_train, y=y_train, verbose=10)

Initialize ModelDriftExplainer and fit it on train and validation data
----------------------------------------------------------------------==

.. code:: python

    >>> from cinnamon.drift import ModelDriftExplainer

    # initialize a drift explainer with the built XGBClassifier and fit it on train
    # and valid data
    >>> drift_explainer = ModelDriftExplainer(model=clf)
    >>> drift_explainer.fit(X1=X_train, X2=X_valid, y1=y_train, y2=y_valid)

Detect data drift by looking at main graphs and metrics
----------------------------------------------------------

.. code:: python

    # Distribution of logit predictions
    >>> drift_explainer.plot_prediction_drift(bins=15)

.. image:: docs/img/plot_prediction_drift.png
    :width: 400
    :align: center

We can see on this graph that because of the data drift we introduced in validation 
data the distribution of predictions are different (they do not overlap well). We 
can also compute the corresponding drift metrics:

.. code:: python

    # Corresponding metrics
    >>> drift_explainer.get_prediction_drift()
    [{'mean_difference': -3.643428434667366,
    'wasserstein': 3.643428434667366,
    'kolmogorov_smirnov': KstestResult(statistic=0.2913775225333014, pvalue=0.00013914094110123454)}]

Comparing the distributions of predictions for two datasets is one of the main 
indicator we use in order to detect data drift. The two other indicators are:

- distribution of the target (see `get_target_drift`)
- performance metrics (see `get_performance_metrics_drift`)

Explain data drift by computing the drift importances
--------------------------------------------------------

Drift importances can be thought as equivalent of feature importances but in terms of data drift.

.. code:: python

    # plot drift values
    >>> drift_explainer.plot_tree_based_drift_importances(n=7)

.. image:: docs/img/plot_drift_values.png
    :width: 400
    :align: center

Here the feature `worst symmetry` is rightly identified as the one which contributes the most to the data drift.

More
------

See "notes" below to explore all the functionalities of CinnaMon.

🛠 Installation
=================

CinnaMon is intended to work with **Python 3.7 or above**. Installation can be done with `pip`:

.. code:: sh
    
    $ pip install cinnamon

🔗 Notes
===========

- The two main classes of CinnaMon are `ModelDriftExplainer` and `AdversarialDriftExplainer`
- `ModelDriftExplainer` currently only support `XGBoost` models (both regression and classification
  are supported)
- See notebooks in the `examples/` directory to have an overview of all functionalities. 
  Notably:

  - `Covariate shift example with IEEE data <https://github.com/zelros/cinnamon/blob/master/examples/ieee_fraud_simulated_covariate_shift_card6.ipynb>`_
  - `Concept drift example with IEEE data <https://github.com/zelros/cinnamon/blob/master/examples/ieee_fraud_simulated_concept_drift_card6.ipynb>`_
    
  These two notebooks also go deeper into the topic of how to correct data drift, making use of `AdversarialDriftExplainer`
- See also the `slide presentation <https://yohannlefaou.github.io/publications/2021-cinnamon/Detect_explain_and_correct_data_drift_in_a_machine_learning_system.pdf>`_ of the CinnaMon library.
- There is (yet) no formal documentation for CinnaMon but docstrings are up to date for the two main classes.


👍 Contributing
=================

Check out the `contribution <https://github.com/zelros/cinnamon/blob/master/CONTRIBUTING.md) section>`_.

📝 License
============

CinnaMon is free and open-source software licensed under the `MIT <https://github.com/zelros/cinnamon/blob/master/LICENSE.txt>`_.
