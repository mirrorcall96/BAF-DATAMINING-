"""Extended Base-only sweep (research-style deep dive).

Sweep axes:
  Imbalance     (6) : none, smote, adasyn, smoteenn, smotetomek, undersample
  Feature sel.  (4) : all, mi, ga, rfe
  Models       (12) : DT, DT-shallow, SVM, ANN, ANN-deep, NB, RF, RF-200,
                       XGBoost, XGBoost-tuned, AdaBoost, HistGB

Total cells: 6 x 4 x 12 = 288 trained models.

Per-cell artifacts saved to outputs/runs/<run_id>.json:
  - config (variant, imbalance, fs, model)
  - metrics (standard 5 + PR-AUC + TPR@5%FPR + threshold-tuned F1+)
  - timings (fit_time_sec, eval_time_sec)
  - n_features, n_train, n_train_pos, n_train_neg

Aggregate parquet at outputs/results_extended.parquet (resumable).

Usage
-----
    python -m src.extended_base               # full sweep (288 cells)
    python -m src.extended_base --skip-done   # default behaviour, explicit
    python -m src.extended_base --limit 50    # only run the first N missing cells
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import pandas as pd

from .config import DUMMY_PREFIXES, OUTPUT_DIR, PREPROCESSED_DIR
from .evaluate import evaluate_extended
from .featsel import select_features
from .imbalance import resample
from .models import build_model

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
EXT_VARIANT = "Base"  # this sweep is Base-only
EXT_IMBALANCE = [
    "none", "smote", "smote_nc", "adasyn", "smoteenn", "smotetomek", "undersample",
]
EXT_FS = ["all", "mi", "ga", "rfe", "mrmr", "boruta", "permutation"]
EXT_MODELS = [
    "DT", "DT-shallow",
    "SVM",
    "ANN", "ANN-deep",
    "NB",
    "RF", "RF-200",
    "XGBoost", "XGBoost-tuned",
    "AdaBoost", "HistGB",
    "LightGBM", "LightGBM-bal",
    "CatBoost",
    "BalancedRF", "EasyEnsemble",
]

EXT_RESULTS = OUTPUT_DIR / "results_extended.parquet"
EXT_FEATURE_CACHE = OUTPUT_DIR / "feature_cache_extended.json"
EXT_RUNS_DIR = OUTPUT_DIR / "runs"
EXT_RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _partition_paths(partition_id: int | None, partition_total: int | None):
    """Per-partition output paths so multiple servers don't write to the same file."""
    if partition_id is None:
        return EXT_RESULTS, EXT_FEATURE_CACHE, EXT_RUNS_DIR
    suffix = f"_partition_{partition_id}_of_{partition_total}"
    res = OUTPUT_DIR / f"results_extended{suffix}.parquet"
    cache = OUTPUT_DIR / f"feature_cache_extended{suffix}.json"
    runs = OUTPUT_DIR / f"runs{suffix}"
    runs.mkdir(parents=True, exist_ok=True)
    return res, cache, runs


def _belongs_to_partition(idx: int, partition_id: int | None, partition_total: int | None) -> bool:
    """Index-based partitioning: cell `idx` belongs to my partition if idx % total == id-1."""
    if partition_id is None:
        return True
    return (idx % partition_total) == (partition_id - 1)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def load_results() -> pd.DataFrame:
    return pd.read_parquet(EXT_RESULTS) if EXT_RESULTS.exists() else pd.DataFrame()


def append_result(row: dict) -> None:
    df_new = pd.DataFrame([row])
    df = pd.concat([load_results(), df_new], ignore_index=True) if EXT_RESULTS.exists() else df_new
    tmp = EXT_RESULTS.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(EXT_RESULTS)


def load_feature_cache() -> dict:
    return json.loads(EXT_FEATURE_CACHE.read_text()) if EXT_FEATURE_CACHE.exists() else {}


def save_feature_cache(cache: dict) -> None:
    EXT_FEATURE_CACHE.write_text(json.dumps(cache, indent=2))


def save_run_artifact(row: dict, selected_cols: list[str]) -> Path:
    """Save the per-model JSON artifact (config + metrics + selected features)."""
    run_id = f"{row['variant']}_{row['imbalance']}_{row['feature_selection']}_{row['model']}"
    artifact = {**row, "selected_features": selected_cols}
    path = EXT_RUNS_DIR / f"{run_id}.json"
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)
    return path


def already_done(results: pd.DataFrame, **kwargs) -> bool:
    if results.empty:
        return False
    mask = pd.Series(True, index=results.index)
    for k, v in kwargs.items():
        mask &= (results[k] == v)
    return bool(mask.any())


# ---------------------------------------------------------------------------
# Per-cell training
# ---------------------------------------------------------------------------
def get_features(cache: dict, variant: str, imbalance: str, fs: str,
                  X_imb, y_imb, all_cols: list[str]) -> list[str]:
    key = f"{variant}|{imbalance}|{fs}"
    if key in cache:
        cols = cache[key]
        if all(c in all_cols for c in cols):
            return cols

    print(f"  computing FS '{fs}' ...", end=" ", flush=True)
    t0 = time.time()
    cols, _ = select_features(X_imb, y_imb, fs)
    print(f"{len(cols)} cols in {time.time()-t0:.1f}s")
    cache[key] = cols
    save_feature_cache(cache)
    return cols


def train_one(variant: str, imbalance: str, fs: str, model_name: str,
              X_train, y_train, X_test, y_test, selected_cols: list[str]) -> dict:
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    X_tr = X_train[selected_cols]
    X_te = X_test[selected_cols]

    t0 = time.time()
    model = build_model(model_name, n_pos=n_pos, n_neg=n_neg, cost_sensitive=False)
    model.fit(X_tr, y_train)
    fit_time = time.time() - t0

    t0 = time.time()
    metrics = evaluate_extended(model, X_te, y_test)
    eval_time = time.time() - t0

    return {
        "variant":           variant,
        "imbalance":         imbalance,
        "feature_selection": fs,
        "model":             model_name,
        "n_features":        len(selected_cols),
        "n_train":           int(len(y_train)),
        "n_train_pos":       n_pos,
        "n_train_neg":       n_neg,
        "fit_time_sec":      round(fit_time, 2),
        "eval_time_sec":     round(eval_time, 2),
        **metrics,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
ROW_HEADER = (
    f"  {'model':<14s} {'fit (s)':>8s} {'AUC':>7s} {'PR-AUC':>7s} {'TPR@5%':>8s} "
    f"{'F1+':>7s} {'F1+(t)':>7s} {'Rec+':>7s} {'Prec+':>7s}"
)


def print_row(row: dict) -> None:
    print(
        f"  {row['model']:<14s} {row['fit_time_sec']:>8.1f} "
        f"{row['auc']:>7.4f} {row['pr_auc']:>7.4f} "
        f"{row['tpr_at_5fpr']:>8.4f} "
        f"{row['f1_pos']:>7.4f} {row['threshold_tuned_f1_pos']:>7.4f} "
        f"{row['recall_pos']:>7.4f} {row['precision_pos']:>7.4f}"
    )


def run(limit: int | None = None,
         partition_id: int | None = None,
         partition_total: int | None = None) -> None:
    # Per-partition output paths
    global EXT_RESULTS, EXT_FEATURE_CACHE, EXT_RUNS_DIR
    EXT_RESULTS, EXT_FEATURE_CACHE, EXT_RUNS_DIR = _partition_paths(partition_id, partition_total)

    feature_cache = load_feature_cache()
    prep_dir = PREPROCESSED_DIR / EXT_VARIANT
    if not prep_dir.exists():
        raise FileNotFoundError(f"Preprocessed Base data not found at {prep_dir} — run prep.py first")

    total_cells = len(EXT_IMBALANCE) * len(EXT_FS) * len(EXT_MODELS)
    if partition_id is not None:
        my_cells = sum(1 for i in range(total_cells) if _belongs_to_partition(i, partition_id, partition_total))
        print("=" * 78)
        print(f"  EXTENDED BASE SWEEP — partition {partition_id}/{partition_total}")
        print(f"  This server runs {my_cells} of {total_cells} cells")
        print(f"  Results → {EXT_RESULTS.name}")
        print("=" * 78)
    else:
        print("=" * 78)
        print(f"  EXTENDED BASE SWEEP  ({len(EXT_IMBALANCE)} x {len(EXT_FS)} x {len(EXT_MODELS)} = "
              f"{total_cells} cells)")
        print("=" * 78)

    X_train = pd.read_parquet(prep_dir / "X_train.parquet")
    y_train = pd.read_parquet(prep_dir / "y_train.parquet").iloc[:, 0]
    X_test = pd.read_parquet(prep_dir / "X_test.parquet")
    y_test = pd.read_parquet(prep_dir / "y_test.parquet").iloc[:, 0]
    all_cols = list(X_train.columns)
    print(f"  train: {X_train.shape}  pos={int(y_train.sum())}")
    print(f"  test : {X_test.shape}  pos={int(y_test.sum())}")

    n_done = 0
    n_run = 0
    n_skipped_partition = 0
    cell_idx = -1

    for imbalance in EXT_IMBALANCE:
        # Skip resample work entirely if no cell of this imbalance belongs to my partition
        my_in_imb = False
        for fs in EXT_FS:
            for model_name in EXT_MODELS:
                cell_idx_check = (
                    EXT_IMBALANCE.index(imbalance) * len(EXT_FS) * len(EXT_MODELS)
                    + EXT_FS.index(fs) * len(EXT_MODELS)
                    + EXT_MODELS.index(model_name)
                )
                if _belongs_to_partition(cell_idx_check, partition_id, partition_total):
                    my_in_imb = True
                    break
            if my_in_imb:
                break
        if not my_in_imb:
            cell_idx += len(EXT_FS) * len(EXT_MODELS)
            continue

        t0 = time.time()
        try:
            X_imb, y_imb = resample(X_train, y_train, imbalance, dummy_prefixes=DUMMY_PREFIXES)
        except Exception as e:
            print(f"\n[!] {imbalance} resample failed: {type(e).__name__}: {e}")
            cell_idx += len(EXT_FS) * len(EXT_MODELS)
            continue
        print(f"\n[{EXT_VARIANT} | {imbalance:11s}] resampled to {X_imb.shape} "
              f"(pos={int(y_imb.sum())}) in {time.time()-t0:.1f}s")

        for fs in EXT_FS:
            # Same FS-skip optimisation
            my_in_fs = any(
                _belongs_to_partition(
                    EXT_IMBALANCE.index(imbalance) * len(EXT_FS) * len(EXT_MODELS)
                    + EXT_FS.index(fs) * len(EXT_MODELS) + i,
                    partition_id, partition_total,
                )
                for i in range(len(EXT_MODELS))
            )
            if not my_in_fs:
                cell_idx += len(EXT_MODELS)
                continue

            try:
                selected = get_features(feature_cache, EXT_VARIANT, imbalance, fs,
                                         X_imb, y_imb, all_cols)
            except Exception as e:
                print(f"  [!] FS {fs} failed: {type(e).__name__}: {e}")
                cell_idx += len(EXT_MODELS)
                continue

            print(f"  [{imbalance:11s} | {fs:4s} | |F|={len(selected)}]")
            print(ROW_HEADER)

            for model_name in EXT_MODELS:
                cell_idx += 1
                if not _belongs_to_partition(cell_idx, partition_id, partition_total):
                    n_skipped_partition += 1
                    continue

                results = load_results()
                if already_done(
                    results, variant=EXT_VARIANT, imbalance=imbalance,
                    feature_selection=fs, model=model_name,
                ):
                    n_done += 1
                    continue

                try:
                    row = train_one(EXT_VARIANT, imbalance, fs, model_name,
                                     X_train=X_imb, y_train=y_imb,
                                     X_test=X_test, y_test=y_test,
                                     selected_cols=selected)
                    append_result(row)
                    save_run_artifact(row, selected)
                    print_row(row)
                    n_run += 1
                    if limit and n_run >= limit:
                        print(f"\n[stopping early — reached --limit {limit}]")
                        return
                except Exception as e:
                    print(f"  {model_name:<14s} [ERR] {type(e).__name__}: {e}")

    print(f"\nFinished. Newly trained: {n_run}. Already done (skipped): {n_done}. "
          f"Other-partition skipped: {n_skipped_partition}.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many newly trained models")
    ap.add_argument("--partition", type=str, default=None,
                    help='partition ID over total servers, e.g. "1/3" = "I am server 1 of 3"')
    args = ap.parse_args()

    partition_id = partition_total = None
    if args.partition:
        partition_id, partition_total = (int(x) for x in args.partition.split("/"))
        assert 1 <= partition_id <= partition_total, "partition must satisfy 1 <= id <= total"

    run(limit=args.limit, partition_id=partition_id, partition_total=partition_total)


if __name__ == "__main__":
    main()
