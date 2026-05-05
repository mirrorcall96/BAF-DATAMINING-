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

import os

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

# Optional imports — wrapped in try/except so the module loads even on
# servers without these libraries. Each model checks availability at build time.
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

try:
    from imblearn.ensemble import (
        BalancedRandomForestClassifier, EasyEnsembleClassifier,
    )
    HAS_IMBLEARN_ENSEMBLE = True
except ImportError:
    HAS_IMBLEARN_ENSEMBLE = False

from .config import (
    HP_ANN, HP_CATBOOST, HP_DT, HP_LGBM, HP_RF, HP_SVM, HP_XGB, NEEDS_SCALING, RANDOM_STATE,
)


def _gpu_available() -> bool:
    """Cheap detection: check for `nvidia-smi` exit code 0."""
    if os.environ.get("FORCE_CPU", "").lower() in {"1", "true", "yes"}:
        return False
    return os.system("nvidia-smi >/dev/null 2>&1") == 0


GPU_AVAILABLE = _gpu_available()


# Extended models for the Base-only deep sweep (`extended_base.py`)
ModelName = Literal[
    "DT", "DT-shallow",
    "SVM",
    "ANN", "ANN-deep",
    "NB",
    "RF", "RF-200",
    "XGBoost", "XGBoost-tuned", "XGBoost-CS",
    "AdaBoost", "HistGB",
    "LightGBM", "LightGBM-bal",
    "CatBoost",
    "BalancedRF", "EasyEnsemble",
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
            n_estimators=200, learning_rate=0.5,
            random_state=RANDOM_STATE,
        )
    if name == "HistGB":
        return HistGradientBoostingClassifier(
            max_iter=300, max_depth=8, learning_rate=0.1,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
            class_weight=("balanced" if cost_sensitive else None),
            random_state=RANDOM_STATE,
        )

    if name in ("LightGBM", "LightGBM-bal"):
        if not HAS_LGBM:
            raise RuntimeError("lightgbm is not installed (`pip install lightgbm`)")
        kwargs = dict(HP_LGBM)
        if name == "LightGBM-bal" or cost_sensitive:
            kwargs["class_weight"] = "balanced"
        if GPU_AVAILABLE:
            kwargs["device"] = "gpu"
            kwargs["gpu_use_dp"] = False
        return LGBMClassifier(**kwargs, random_state=RANDOM_STATE)

    if name == "CatBoost":
        if not HAS_CATBOOST:
            raise RuntimeError("catboost is not installed (`pip install catboost`)")
        kwargs = dict(HP_CATBOOST)
        kwargs["auto_class_weights"] = "Balanced" if cost_sensitive else None
        if GPU_AVAILABLE:
            kwargs["task_type"] = "GPU"
            kwargs["devices"] = "0"
        return CatBoostClassifier(**kwargs, random_state=RANDOM_STATE)

    if name == "BalancedRF":
        if not HAS_IMBLEARN_ENSEMBLE:
            raise RuntimeError("imbalanced-learn ensemble module not available")
        return BalancedRandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=10,
            sampling_strategy="not minority", replacement=True, bootstrap=False,
            n_jobs=-1, random_state=RANDOM_STATE,
        )

    if name == "EasyEnsemble":
        if not HAS_IMBLEARN_ENSEMBLE:
            raise RuntimeError("imbalanced-learn ensemble module not available")
        return EasyEnsembleClassifier(
            n_estimators=10,
            estimator=AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
                n_estimators=100, learning_rate=0.5, random_state=RANDOM_STATE,
            ),
            n_jobs=-1, random_state=RANDOM_STATE,
        )

    raise ValueError(f"Unknown model: {name!r}")


def build_model(name: str, n_pos: int, n_neg: int, cost_sensitive: bool = False):
    """Return a (possibly Pipeline-wrapped) sklearn-compatible classifier."""
    est = _make_estimator(name, n_pos=n_pos, n_neg=n_neg, cost_sensitive=cost_sensitive)
    if name in EXTENDED_NEEDS_SCALING:
        # GaussianNB is fine without mean-centring; SVM/ANN benefit from it
        return Pipeline([("scaler", StandardScaler(with_mean=(name != "NB"))), ("clf", est)])
    return est
