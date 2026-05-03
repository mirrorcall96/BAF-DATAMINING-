"""Step 1 — Data preparation.

A single shared pipeline applied to all 6 BAF variants:
  1. Load CSV
  2. Decode -1 sentinels to NaN in 5 known columns
  3. Add `<col>_was_missing` indicator columns
  4. Median imputation (numeric) / mode imputation (categorical)
  5. IQR-based clipping (winsorisation) on heavy-tailed numeric cols
  6. Drop constant columns (e.g. device_fraud_count which is always 0)
  7. One-hot encode 5 categorical columns
  8. Drop multicollinear features (|corr| > CORR_DROP_THRESHOLD)
  9. Temporal split: months 0-5 -> train, months 6-7 -> test
 10. Save preprocessed parquet (X_train, y_train, X_test, y_test, prep_report)

Standardisation (Z-score) is NOT applied here — it is injected per-model in
`models.py` for SVM/ANN/NB only (trees ignore scaling).
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np
import pandas as pd

from .config import (
    DATA_DIR, PREPROCESSED_DIR, VARIANTS, TARGET, MONTH_COL,
    SENTINEL_NEG1, CATEGORICAL, CLIP_COLUMNS, CORR_DROP_THRESHOLD,
    TRAIN_MONTHS, TEST_MONTHS,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------
def decode_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace -1 with NaN in sentinel columns; add a missing-indicator column."""
    for col in SENTINEL_NEG1:
        if col in df.columns:
            mask = df[col] == -1
            df[f"{col}_was_missing"] = mask.astype(np.int8)
            df.loc[mask, col] = np.nan
    return df


def impute(df: pd.DataFrame, train_idx: pd.Index) -> tuple[pd.DataFrame, dict]:
    """Median (numeric) / mode (categorical) imputation. Stats from train only."""
    fill_values: dict = {}
    train = df.loc[train_idx]
    for col in df.columns:
        if df[col].isna().any():
            if df[col].dtype.kind in "if":
                fill = float(train[col].median())
            else:
                mode = train[col].mode()
                fill = mode.iloc[0] if len(mode) else ""
            df[col] = df[col].fillna(fill)
            fill_values[col] = fill
    return df, fill_values


def iqr_clip(df: pd.DataFrame, train_idx: pd.Index) -> tuple[pd.DataFrame, dict]:
    """IQR winsorisation on heavy-tailed numerics (1.5 IQR rule, train-fit)."""
    bounds: dict = {}
    train = df.loc[train_idx]
    for col in CLIP_COLUMNS:
        if col not in df.columns:
            continue
        q1 = float(train[col].quantile(0.25))
        q3 = float(train[col].quantile(0.75))
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        bounds[col] = {
            "q1": q1, "q3": q3, "lo": lo, "hi": hi,
            "n_clipped_low":  int((df[col] < lo).sum()),
            "n_clipped_high": int((df[col] > hi).sum()),
        }
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df, bounds


def one_hot_encode(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode the 5 categorical columns."""
    present = [c for c in CATEGORICAL if c in df.columns]
    encoded = pd.get_dummies(df, columns=present, prefix=present, dtype=np.int8)
    new_cols = [c for c in encoded.columns if any(c.startswith(p + "_") for p in present)]
    return encoded, new_cols


def drop_multicollinear(df: pd.DataFrame, train_idx: pd.Index, target: str,
                          threshold: float = CORR_DROP_THRESHOLD) -> tuple[pd.DataFrame, list[str]]:
    """Drop one of every pair of features with |Pearson rho| > threshold (train-fit)."""
    features = [c for c in df.columns if c != target]
    train_feats = df.loc[train_idx, features]
    corr = train_feats.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    if to_drop:
        df = df.drop(columns=to_drop)
    return df, to_drop


def temporal_split(df: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """Months 0-5 -> train, 6-7 -> test."""
    train_idx = df.index[df[MONTH_COL].isin(TRAIN_MONTHS)]
    test_idx = df.index[df[MONTH_COL].isin(TEST_MONTHS)]
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Per-variant orchestration
# ---------------------------------------------------------------------------
def prep_one_variant(name: str, csv_path) -> dict:
    """Apply the full pipeline to one CSV; save outputs; return a report dict."""
    t0 = time.time()
    print(f"[{name}] loading {csv_path.name} ...")
    df = pd.read_csv(csv_path)

    train_idx, test_idx = temporal_split(df)
    print(f"[{name}] train rows: {len(train_idx):,} | test rows: {len(test_idx):,}")

    df = decode_sentinels(df)
    df, fill_values = impute(df, train_idx)
    df, clip_bounds = iqr_clip(df, train_idx)

    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        df = df.drop(columns=constant_cols)
        print(f"[{name}] dropped {len(constant_cols)} constant col(s): {constant_cols}")

    df, ohe_cols = one_hot_encode(df)
    df, dropped_corr = drop_multicollinear(df, train_idx, TARGET)
    if dropped_corr:
        print(f"[{name}] dropped {len(dropped_corr)} multicollinear col(s): {dropped_corr}")

    feature_cols = [c for c in df.columns if c not in (TARGET, MONTH_COL)]
    X_train = df.loc[train_idx, feature_cols].astype(np.float32)
    y_train = df.loc[train_idx, TARGET].astype(np.int8)
    X_test = df.loc[test_idx, feature_cols].astype(np.float32)
    y_test = df.loc[test_idx, TARGET].astype(np.int8)

    out_dir = PREPROCESSED_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(out_dir / "X_train.parquet", index=False)
    y_train.to_frame().to_parquet(out_dir / "y_train.parquet", index=False)
    X_test.to_parquet(out_dir / "X_test.parquet", index=False)
    y_test.to_frame().to_parquet(out_dir / "y_test.parquet", index=False)

    elapsed = time.time() - t0
    report = {
        "variant": name,
        "raw_rows": int(df.shape[0]),
        "n_features_final": len(feature_cols),
        "feature_cols": feature_cols,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_train_positive": int(y_train.sum()),
        "n_test_positive": int(y_test.sum()),
        "train_prevalence_pct": round(float(y_train.mean()) * 100, 4),
        "test_prevalence_pct": round(float(y_test.mean()) * 100, 4),
        "constant_cols_dropped": constant_cols,
        "multicollinear_cols_dropped": dropped_corr,
        "imputation_fill_values": fill_values,
        "iqr_clip_bounds": clip_bounds,
        "one_hot_columns": ohe_cols,
        "elapsed_sec": round(elapsed, 2),
    }
    with open(out_dir / "prep_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"[{name}] done in {elapsed:.1f}s | features={len(feature_cols)} | "
          f"train pos={y_train.sum():,} ({y_train.mean()*100:.3f}%) | "
          f"test pos={y_test.sum():,} ({y_test.mean()*100:.3f}%)")
    return report


def main() -> None:
    all_reports = {}
    for name, fname in VARIANTS.items():
        all_reports[name] = prep_one_variant(name, DATA_DIR / fname)

    print("\n" + "=" * 78)
    print("PREP SUMMARY")
    print("=" * 78)
    print(f"{'variant':<12} {'feats':>6} {'train':>10} {'test':>10} "
          f"{'tr_pos':>8} {'te_pos':>8} {'tr_prev%':>9} {'te_prev%':>9}")
    for name, r in all_reports.items():
        print(f"{name:<12} {r['n_features_final']:>6} {r['n_train']:>10,} {r['n_test']:>10,} "
              f"{r['n_train_positive']:>8,} {r['n_test_positive']:>8,} "
              f"{r['train_prevalence_pct']:>9.3f} {r['test_prevalence_pct']:>9.3f}")

    summary_path = PREPROCESSED_DIR / "ALL_prep_reports.json"
    with open(summary_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
