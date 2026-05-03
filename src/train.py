"""Step 6-7 — Training orchestrator.

Loops over (variant, imbalance, feature-selection, model), prints each model's
result as soon as it finishes, and persists results incrementally to
`outputs/results.parquet`. Resumable: skips combinations already in the parquet.

Usage
-----
    # default: all 6 variants, all axes, plus cost-sensitive XGBoost per variant
    python -m src.train

    # subset of variants
    python -m src.train --variants Base VariantI

    # skip the cost-sensitive run
    python -m src.train --skip-cs

    # only do the cost-sensitive run (after baseline is complete)
    python -m src.train --only-cs
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import pandas as pd

from .config import (
    COST_SENSITIVE_MODEL, DUMMY_PREFIXES, FEATURE_CACHE_PATH, FS_METHODS,
    IMBALANCE_METHODS, MODELS, PREPROCESSED_DIR, RESULTS_PATH, VARIANTS,
)
from .evaluate import evaluate
from .featsel import select_features
from .imbalance import resample
from .models import build_model

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def load_results() -> pd.DataFrame:
    return pd.read_parquet(RESULTS_PATH) if RESULTS_PATH.exists() else pd.DataFrame()


def append_result(row: dict) -> None:
    """Atomically append one row to results.parquet."""
    df_new = pd.DataFrame([row])
    df = pd.concat([load_results(), df_new], ignore_index=True) if RESULTS_PATH.exists() else df_new
    tmp = RESULTS_PATH.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(RESULTS_PATH)


def load_feature_cache() -> dict:
    return json.loads(FEATURE_CACHE_PATH.read_text()) if FEATURE_CACHE_PATH.exists() else {}


def save_feature_cache(cache: dict) -> None:
    FEATURE_CACHE_PATH.write_text(json.dumps(cache, indent=2))


def already_done(results: pd.DataFrame, **kwargs) -> bool:
    if results.empty:
        return False
    mask = pd.Series(True, index=results.index)
    for k, v in kwargs.items():
        mask &= (results[k] == v)
    return bool(mask.any())


# ---------------------------------------------------------------------------
# One-model training
# ---------------------------------------------------------------------------
def train_one_model(
    variant: str, imbalance: str, fs: str, model_name: str, cost_sensitive: bool,
    X_train, y_train, X_test, y_test, selected_cols: list[str],
) -> dict:
    """Fit one model and return a result row (metrics + bookkeeping)."""
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    X_tr = X_train[selected_cols]
    X_te = X_test[selected_cols]

    t0 = time.time()
    model = build_model(model_name, n_pos=n_pos, n_neg=n_neg, cost_sensitive=cost_sensitive)
    model.fit(X_tr, y_train)
    fit_time = time.time() - t0

    t0 = time.time()
    metrics = evaluate(model, X_te, y_test)
    eval_time = time.time() - t0

    return {
        "variant":           variant,
        "imbalance":         imbalance,
        "feature_selection": fs,
        "model":             model_name + ("-CS" if cost_sensitive else ""),
        "cost_sensitive":    cost_sensitive,
        "n_features":        len(selected_cols),
        "n_train":           int(len(y_train)),
        "n_train_pos":       n_pos,
        "n_train_neg":       n_neg,
        "fit_time_sec":      round(fit_time, 2),
        "eval_time_sec":     round(eval_time, 2),
        **metrics,
    }


def get_features(
    cache: dict, variant: str, imbalance: str, fs: str,
    X_imb, y_imb, all_cols: list[str],
) -> list[str]:
    """Look up FS result in cache, recompute and store if absent."""
    key = f"{variant}|{imbalance}|{fs}"
    if key in cache:
        cols = cache[key]
        if all(c in all_cols for c in cols):
            return cols
        print(f"  cache hit but stale columns — recomputing FS")

    print(f"  computing FS '{fs}' for ({variant}, {imbalance}) ...", end=" ", flush=True)
    t0 = time.time()
    cols, _ = select_features(X_imb, y_imb, fs)
    print(f"{len(cols)} cols in {time.time()-t0:.1f}s")
    cache[key] = cols
    save_feature_cache(cache)
    return cols


# ---------------------------------------------------------------------------
# Live progress printing
# ---------------------------------------------------------------------------
ROW_HEADER = (
    f"  {'model':<11s} {'fit (s)':>8s} {'AUC':>7s} {'F1+':>7s} "
    f"{'Rec+':>7s} {'Prec+':>7s} {'Acc':>7s}"
)


def print_row(row: dict) -> None:
    print(
        f"  {row['model']:<11s} {row['fit_time_sec']:>8.1f} "
        f"{row['auc']:>7.4f} {row['f1_pos']:>7.4f} "
        f"{row['recall_pos']:>7.4f} {row['precision_pos']:>7.4f} "
        f"{row['accuracy']:>7.4f}"
    )


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def run(variants: list[str], skip_cs: bool = False, only_cs: bool = False) -> None:
    feature_cache = load_feature_cache()

    for variant in variants:
        prep_dir = PREPROCESSED_DIR / variant
        if not prep_dir.exists():
            print(f"[!] No preprocessed data for {variant} — run prep.py first")
            continue

        print("\n" + "=" * 78)
        print(f"  {variant}")
        print("=" * 78)
        X_train = pd.read_parquet(prep_dir / "X_train.parquet")
        y_train = pd.read_parquet(prep_dir / "y_train.parquet").iloc[:, 0]
        X_test = pd.read_parquet(prep_dir / "X_test.parquet")
        y_test = pd.read_parquet(prep_dir / "y_test.parquet").iloc[:, 0]
        all_cols = list(X_train.columns)
        print(f"  train: {X_train.shape}  pos={int(y_train.sum())}")
        print(f"  test : {X_test.shape}  pos={int(y_test.sum())}")

        if not only_cs:
            for imbalance in IMBALANCE_METHODS:
                t0 = time.time()
                X_imb, y_imb = resample(X_train, y_train, imbalance, dummy_prefixes=DUMMY_PREFIXES)
                print(f"\n[{variant} | {imbalance}] resampled to {X_imb.shape} "
                      f"(pos={int(y_imb.sum())}) in {time.time()-t0:.1f}s")

                for fs in FS_METHODS:
                    selected = get_features(feature_cache, variant, imbalance, fs,
                                            X_imb, y_imb, all_cols)
                    print(ROW_HEADER)
                    for model_name in MODELS:
                        if already_done(load_results(), variant=variant, imbalance=imbalance,
                                         feature_selection=fs, model=model_name):
                            print(f"  {model_name:<11s} (skipped — already in results)")
                            continue
                        try:
                            row = train_one_model(
                                variant, imbalance, fs, model_name, cost_sensitive=False,
                                X_train=X_imb, y_train=y_imb,
                                X_test=X_test, y_test=y_test, selected_cols=selected,
                            )
                            append_result(row)
                            print_row(row)
                        except Exception as e:
                            print(f"  [ERR] {model_name}: {type(e).__name__}: {e}")

        if not skip_cs:
            results = load_results()
            cs_label = COST_SENSITIVE_MODEL + "-CS"
            if already_done(results, variant=variant, model=cs_label):
                print(f"\n[{variant}] cost-sensitive {COST_SENSITIVE_MODEL} already done")
                continue

            base_rows = results[(results["variant"] == variant) & (results["model"] == COST_SENSITIVE_MODEL)]
            if base_rows.empty:
                best_imb, best_fs = "none", "all"
            else:
                best = base_rows.sort_values("f1_pos", ascending=False).iloc[0]
                best_imb, best_fs = best["imbalance"], best["feature_selection"]

            print(f"\n[{variant} | cost-sensitive] best baseline config: "
                  f"imbalance={best_imb}, fs={best_fs}")
            X_imb, y_imb = resample(X_train, y_train, best_imb, dummy_prefixes=DUMMY_PREFIXES)
            selected = get_features(feature_cache, variant, best_imb, best_fs,
                                     X_imb, y_imb, all_cols)
            print(ROW_HEADER)
            row = train_one_model(
                variant, best_imb, best_fs, COST_SENSITIVE_MODEL, cost_sensitive=True,
                X_train=X_imb, y_train=y_imb,
                X_test=X_test, y_test=y_test, selected_cols=selected,
            )
            append_result(row)
            print_row(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS.keys()),
                    help="variants to run (default: all 6)")
    ap.add_argument("--skip-cs", action="store_true", help="skip the cost-sensitive run")
    ap.add_argument("--only-cs", action="store_true", help="only run cost-sensitive (after baseline)")
    args = ap.parse_args()
    run(args.variants, skip_cs=args.skip_cs, only_cs=args.only_cs)


if __name__ == "__main__":
    main()
