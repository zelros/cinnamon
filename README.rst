.. -*- mode: rst -*-

|ReadTheDocs|_ |License|_ |PyPi|_

.. |ReadTheDocs| image:: https://readthedocs.org/projects/cinnamon/badge
.. _ReadTheDocs: https://cinnamon.readthedocs.io/en/add-documentation

.. |License| image:: https://img.shields.io/badge/License-MIT-yellow
.. _License: https://github.com/zelros/cinnamon/blob/master/LICENSE.txt

.. |PyPi| image:: https://img.shields.io/pypi/v/cinnamon
.. _PyPi: https://pypi.org/project/cinnamon/


CinnaMon
====================================================

**CinnaMon** is a Python library which allows to monitor data drift on a 
machine learning system. It provides tools to study data drift between two datasets,
especially to detect, explain, and correct data drift.

⚡️ Quickstart
==============

As a quick example, let's illustrate the use of CinnaMon on the breast cancer data where we voluntarily introduce some data drift.

Setup the data and build a model

.. code:: python

    import pandas as pd
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    # load breast cancer data
    dataset = datasets.load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns = dataset.feature_names)
    y = dataset.target

    # split data in train and valid dataset
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=2021)

    # introduce some data drift in valid by filtering with 'worst symmetry' feature
    y_valid = y_valid[X_valid['worst symmetry'].values > 0.3]
    X_valid = X_valid.loc[X_valid['worst symmetry'].values > 0.3, :].copy()

    # fit a XGBClassifier on the training data
    clf = XGBClassifier(use_label_encoder=False)
    clf.fit(X=X_train, y=y_train, verbose=10)

Initialize ModelDriftExplainer and fit it on train and validation data

.. code:: python

    from cinnamon.drift import ModelDriftExplainer

    # initialize a drift explainer with the built XGBClassifier and fit it on train
    # and valid data
    drift_explainer = ModelDriftExplainer(model=clf)
    drift_explainer.fit(X1=X_train, X2=X_valid, y1=y_train, y2=y_valid)

Detect data drift by looking at main graphs and metrics

.. code:: python
    
    # Distribution of logit predictions
    drift_explainer.plot_prediction_drift(bins=15)
