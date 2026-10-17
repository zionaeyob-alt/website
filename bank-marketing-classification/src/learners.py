"""Custom FLAML learner: logistic regression with one-hot encoding and scaling.

FLAML's built-in `lrl1`/`lrl2` learners turn categories into integer codes and do
not scale features, which handicaps a linear model. This learner one-hot encodes
categoricals and standardises numerics inside the model, so it can be tuned
by FLAML like any other estimator and pickled with the AutoML object.
"""
import numpy as np
import pandas as pd
from flaml import tune
from flaml.automl.model import SKLearnEstimator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class OneHotLogisticRegression(ClassifierMixin, BaseEstimator):
    def __init__(self, C=1.0, max_iter=2000, random_state=None, n_jobs=None):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, **kwargs):
        X = pd.DataFrame(X)
        cat_cols = X.select_dtypes(exclude="number").columns.tolist()
        num_cols = [c for c in X.columns if c not in cat_cols]
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ])
        clf = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=self.random_state)
        self.pipeline_ = Pipeline([("pre", pre), ("clf", clf)]).fit(X, y)
        self.classes_ = self.pipeline_.classes_
        return self

    def predict_proba(self, X):
        return self.pipeline_.predict_proba(pd.DataFrame(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class ScaledLogisticRegression(SKLearnEstimator):
    """FLAML wrapper; registered in AutoML as the learner named "lr"."""

    @classmethod
    def search_space(cls, data_size, task, **params):
        return {"C": {"domain": tune.loguniform(lower=1e-3, upper=1e3), "init_value": 1.0}}

    @classmethod
    def cost_relative2lgbm(cls):
        return 20

    def __init__(self, task="binary", **config):
        super().__init__(task, **config)
        self.params.pop("random_seed", None)
        self.estimator_class = OneHotLogisticRegression

    def _preprocess(self, X):
        return X  # keep categorical dtypes; the pipeline encodes them
