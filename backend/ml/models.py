"""Model registry. All models predict the h-day-ahead log-return; prices are reconstructed afterwards.

Hyper-parameters are deliberately conservative: daily returns have a very low signal-to-noise ratio and
flexible models overfit easily. Preprocessing (scaling) lives inside each sklearn Pipeline so it is fitted
on training rows only.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from backend.ml.features import FEATURE_COLUMNS


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """'Tomorrow equals today': predicted return is 0."""
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class DriftBaseline(BaseEstimator, RegressorMixin):
    """Predicts the mean historical return over the training window."""
    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


class MeanReversionBaseline(BaseEstimator, RegressorMixin):
    """Predicts that price moves back to its 20-day simple moving average."""
    def fit(self, X, y):
        self.idx_ = FEATURE_COLUMNS.index("dev_sma20")
        return self

    def predict(self, X):
        X = np.asarray(X)
        return -X[:, self.idx_]


MODELS: dict[str, dict] = {
    "knn": {"label": "K-Nearest Neighbors", "kind": "ml", "seed_from": "KNN notebook",
            "description": "Averages the outcomes of the 30 most similar past market states (standardized features).",
            "factory": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=30, weights="distance"))},
    "random_forest": {"label": "Random Forest", "kind": "ml", "seed_from": "RF notebook",
            "description": "200 shallow decision trees with large leaves, averaged to limit overfitting.",
            "factory": lambda: RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=25, max_features=0.5, n_jobs=-1, random_state=42)},
    "bayesian_ridge": {"label": "Bayesian Ridge", "kind": "ml", "seed_from": "Bayesian notebook",
            "description": "Linear model with learned regularization; stable and hard to overfit.",
            "factory": lambda: make_pipeline(StandardScaler(), BayesianRidge())},
    "decision_tree": {"label": "Decision Tree", "kind": "ml", "seed_from": "Decision-tree notebook",
            "description": "A single shallow tree (depth 4, leaf >= 30 samples). Interpretable but high-variance.",
            "factory": lambda: DecisionTreeRegressor(max_depth=4, min_samples_leaf=30, random_state=42)},
    "gradient_boosting": {"label": "Gradient Boosting", "kind": "ml", "seed_from": "Gradient-boosting notebook",
            "description": "150 very shallow trees fitted sequentially with subsampling and a small learning rate.",
            "factory": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42)},
    "svr": {"label": "Support Vector Regression", "kind": "ml", "seed_from": "SVR notebook",
            "description": "RBF-kernel SVR on standardized features and a standardized target.",
            "factory": lambda: TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), SVR(kernel="rbf", C=0.5, epsilon=0.2)), transformer=StandardScaler())},
    "persistence": {"label": "Baseline: Persistence", "kind": "baseline",
            "description": "Naive forecast: the price stays where it is. Any useful model must beat this.",
            "factory": PersistenceBaseline},
    "drift": {"label": "Baseline: Historical Drift", "kind": "baseline",
            "description": "Naive forecast: price grows at its average historical rate.",
            "factory": DriftBaseline},
    "mean_reversion_ma20": {"label": "Baseline: Return to 20-day MA", "kind": "baseline",
            "description": "Moving-average forecast: price moves back to its 20-day average.",
            "factory": MeanReversionBaseline},
}


def get_model(name: str):
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODELS)}")
    return MODELS[name]["factory"]()


def catalog() -> list[dict]:
    return [{"id": k, "label": v["label"], "kind": v["kind"], "description": v["description"]} for k, v in MODELS.items()]
