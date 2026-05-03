"""Step 4-5 — Classifier factory.

Six classifiers from the lectures, plus a cost-sensitive XGBoost variant:
  - DT          : Decision Tree (CART/Gini)
  - SVM         : LinearSVC
  - ANN         : MLPClassifier (small)
  - NB          : GaussianNB on standardised features
  - RF          : Random Forest (bagging)
  - XGBoost     : XGBClassifier (gradient boosting)
  - XGBoost-CS  : Cost-sensitive XGBoost (scale_pos_weight = n_neg / n_pos)

Models that need feature scaling (SVM/ANN/NB) are wrapped in a Pipeline with
StandardScaler so the calling code does not need to scale separately.
All hyperparameters live in `config.py`.
"""
from __future__ import annotations

from typing import Literal

from sklearn.ensemble import (
    AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import (
    HP_ANN, HP_DT, HP_RF, HP_SVM, HP_XGB, NEEDS_SCALING, RANDOM_STATE,
)

# Extended models for the Base-only deep sweep (`extended_base.py`)
ModelName = Literal[
    "DT", "DT-shallow",
    "SVM",
    "ANN", "ANN-deep",
    "NB",
    "RF", "RF-200",
    "XGBoost", "XGBoost-tuned", "XGBoost-CS",
    "AdaBoost", "HistGB",
]
EXTENDED_NEEDS_SCALING = NEEDS_SCALING | {"ANN-deep"}


def _make_estimator(name: str, n_pos: int, n_neg: int, cost_sensitive: bool):
    """Bare estimator (no scaling)."""
    if name == "DT":
        return DecisionTreeClassifier(
            **HP_DT, random_state=RANDOM_STATE,
            class_weight=("balanced" if cost_sensitive else None),
        )
    if name == "DT-shallow":
        return DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20, random_state=RANDOM_STATE,
            class_weight=("balanced" if cost_sensitive else None),
        )
    if name == "SVM":
        return LinearSVC(
            **HP_SVM, random_state=RANDOM_STATE,
            class_weight=("balanced" if cost_sensitive else None),
        )
    if name == "ANN":
        return MLPClassifier(**HP_ANN, random_state=RANDOM_STATE)
    if name == "ANN-deep":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
            learning_rate_init=1e-3, batch_size=512, max_iter=40,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=5,
            random_state=RANDOM_STATE,
        )
    if name == "NB":
        return GaussianNB()
    if name == "RF":
        return RandomForestClassifier(
            **HP_RF, random_state=RANDOM_STATE,
            class_weight=("balanced" if cost_sensitive else None),
        )
    if name == "RF-200":
        return RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=10,
            n_jobs=-1, random_state=RANDOM_STATE,
            class_weight=("balanced" if cost_sensitive else None),
        )
    if name in ("XGBoost", "XGBoost-CS"):
        spw = (n_neg / max(1, n_pos)) if (cost_sensitive or name == "XGBoost-CS") else 1.0
        return XGBClassifier(**HP_XGB, random_state=RANDOM_STATE, scale_pos_weight=spw)
    if name == "XGBoost-tuned":
        spw = (n_neg / max(1, n_pos)) if cost_sensitive else 1.0
        return XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            tree_method="hist", n_jobs=-1, eval_metric="auc", verbosity=0,
            random_state=RANDOM_STATE, scale_pos_weight=spw,
        )
    if name == "AdaBoost":
        # Newer sklearn (>=1.6) dropped the `algorithm` kwarg; SAMME is now the only option
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
            n_estimators=100, learning_rate=0.5,
            random_state=RANDOM_STATE,
        )
    if name == "HistGB":
        return HistGradientBoostingClassifier(
            max_iter=200, max_depth=8, learning_rate=0.1,
            class_weight=("balanced" if cost_sensitive else None),
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unknown model: {name!r}")


def build_model(name: str, n_pos: int, n_neg: int, cost_sensitive: bool = False):
    """Return a (possibly Pipeline-wrapped) sklearn-compatible classifier."""
    est = _make_estimator(name, n_pos=n_pos, n_neg=n_neg, cost_sensitive=cost_sensitive)
    if name in EXTENDED_NEEDS_SCALING:
        # GaussianNB is fine without mean-centring; SVM/ANN benefit from it
        return Pipeline([("scaler", StandardScaler(with_mean=(name != "NB"))), ("clf", est)])
    return est
