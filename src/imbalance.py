"""Step 2 — Imbalance handling.

Three strategies (Lecture 6), applied to the TRAINING set only (never test):
  - "none"        : no resampling (baseline)
  - "smote"       : Synthetic Minority Over-Sampling Technique
  - "undersample" : random undersampling of the majority class

Cost-sensitive learning is treated separately at the model level (see models.py).

Note on SMOTE + one-hot dummies: SMOTE interpolates between minority neighbours
which produces non-binary values for one-hot columns. We round those columns
back to {0, 1} so the data stays interpretable for distance/probability models.
Tree models are insensitive either way.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC
from imblearn.under_sampling import RandomUnderSampler

from .config import DUMMY_PREFIXES, RANDOM_STATE

ImbalanceMethod = Literal[
    "none", "smote", "smote_nc", "adasyn", "smoteenn", "smotetomek", "undersample",
]


# Heuristic: a column is "categorical" for SMOTE-NC if its name starts with one of
# the dummy prefixes (one-hot output) OR ends with the missing-indicator suffix OR
# is one of the documented binary-flag columns in BAF.
BINARY_FLAGS = {
    "has_other_cards", "foreign_request", "email_is_free",
    "phone_home_valid", "phone_mobile_valid", "keep_alive_session",
}


def _categorical_indices(X: pd.DataFrame) -> list[int]:
    cat = []
    for i, col in enumerate(X.columns):
        if col in BINARY_FLAGS:
            cat.append(i)
        elif col.endswith("_was_missing"):
            cat.append(i)
        elif any(col.startswith(p + "_") for p in DUMMY_PREFIXES):
            cat.append(i)
    return cat


def _round_one_hot(X: pd.DataFrame, dummy_prefixes: list[str]) -> pd.DataFrame:
    """Round dummy-prefixed columns back to {0, 1} after SMOTE interpolation."""
    for col in X.columns:
        if any(col.startswith(p + "_") for p in dummy_prefixes):
            X[col] = (X[col] >= 0.5).astype(np.int8)
    return X


def resample(
    X: pd.DataFrame,
    y: pd.Series,
    method: ImbalanceMethod,
    dummy_prefixes: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the requested resampling method to (X, y) and return the result."""
    if method == "none":
        return X, y

    if method == "smote":
        sampler = SMOTE(random_state=RANDOM_STATE)
    elif method == "smote_nc":
        cat_idx = _categorical_indices(X)
        sampler = SMOTENC(categorical_features=cat_idx, random_state=RANDOM_STATE)
    elif method == "adasyn":
        sampler = ADASYN(random_state=RANDOM_STATE)
    elif method == "smoteenn":
        sampler = SMOTEENN(random_state=RANDOM_STATE)
    elif method == "smotetomek":
        sampler = SMOTETomek(random_state=RANDOM_STATE)
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown imbalance method: {method!r}")

    X_res, y_res = sampler.fit_resample(X, y)
    # SMOTE-family that interpolates can produce non-binary dummy values — round them back.
    # SMOTE-NC handles categoricals natively by mode-vote, so no rounding needed.
    if dummy_prefixes and method in ("smote", "adasyn", "smoteenn", "smotetomek"):
        X_res = _round_one_hot(X_res, dummy_prefixes)
    return X_res, y_res


def class_summary(X: pd.DataFrame, y: pd.Series) -> dict:
    """Small dict describing the class balance — useful for logging."""
    counts = y.value_counts().to_dict()
    return {
        "n_total":      int(len(y)),
        "n_negative":   int(counts.get(0, 0)),
        "n_positive":   int(counts.get(1, 0)),
        "positive_pct": round(float(y.mean()) * 100, 4),
        "n_features":   int(X.shape[1]),
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    from .config import PREPROCESSED_DIR

    X = pd.read_parquet(PREPROCESSED_DIR / "Base" / "X_train.parquet")
    y = pd.read_parquet(PREPROCESSED_DIR / "Base" / "y_train.parquet").iloc[:, 0]

    print(f"Original   : {class_summary(X, y)}")
    for method in ("none", "smote", "undersample"):
        t0 = time.time()
        Xr, yr = resample(X, y, method, dummy_prefixes=DUMMY_PREFIXES)
        print(f"{method:11s}: {class_summary(Xr, yr)}  ({time.time()-t0:.1f}s)")
