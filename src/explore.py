"""Quick exploration of the BAF suite — Base + 5 variants.

Prints concise summaries to stdout and writes a JSON report to outputs/exploration.json.
"""
import json
import warnings
from pathlib import Path

import pandas as pd

from .config import DATA_DIR as DATA, OUTPUT_DIR as OUT, SENTINEL_NEG1, VARIANTS as FILES

warnings.filterwarnings("ignore")


def summarize(name: str, path: Path) -> dict:
    print(f"\n{'='*72}\n  {name}  ({path.name})\n{'='*72}")
    df = pd.read_csv(path)

    n_rows, n_cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2

    # Dtypes
    dtypes = df.dtypes.astype(str).value_counts().to_dict()

    # Target
    target = df["fraud_bool"]
    pos = int(target.sum())
    neg = int((target == 0).sum())
    prev = pos / len(target)

    # Per-month structure
    month_counts = df["month"].value_counts().sort_index().to_dict()
    fraud_per_month = df.groupby("month")["fraud_bool"].mean().round(5).to_dict()
    n_per_month = df.groupby("month").size().to_dict()

    # Identify categorical columns (object dtype) and their cardinalities
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_card = {c: int(df[c].nunique()) for c in cat_cols}
    cat_top = {c: df[c].value_counts().head(5).to_dict() for c in cat_cols}

    # Missing values (true NaN, not sentinel)
    missing = df.isna().sum()
    missing = missing[missing > 0].to_dict()

    # Sentinel -1 frequencies
    sentinel_pct = {}
    for c in SENTINEL_NEG1:
        if c in df.columns:
            sentinel_pct[c] = round((df[c] == -1).mean() * 100, 3)

    # Numeric stats summary (just describe)
    num = df.select_dtypes(include="number")
    skew = num.skew(numeric_only=True).round(3).to_dict()
    nunique = {c: int(df[c].nunique()) for c in num.columns}

    # Top correlations with target
    corr = num.corr()["fraud_bool"].drop("fraud_bool").abs().sort_values(ascending=False)
    top_corr = corr.head(10).round(4).to_dict()

    # Duplicates check (full row, expensive — sample only)
    dup_pct_sample = float(df.sample(min(200_000, n_rows), random_state=0).duplicated().mean() * 100)

    print(f"  Shape: {n_rows:,} rows × {n_cols} cols   |   memory {mem_mb:.1f} MB")
    print(f"  Dtypes: {dtypes}")
    print(f"  Target: positive={pos:,} ({prev*100:.3f}%), negative={neg:,}")
    print(f"  Months present: {sorted(month_counts.keys())}")
    print(f"  Rows per month: {n_per_month}")
    print(f"  Fraud rate per month: {fraud_per_month}")
    print(f"  Categorical cardinalities: {cat_card}")
    print(f"  Columns with NaN: {missing if missing else 'none'}")
    print(f"  Sentinel -1 percent: {sentinel_pct}")
    print(f"  Top |corr| with target: {top_corr}")
    print(f"  Top-skew numeric features:")
    top_skew = sorted(skew.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    for k, v in top_skew:
        print(f"     {k:35s}  skew={v:+.2f}")

    return {
        "name": name,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "memory_mb": round(mem_mb, 2),
        "dtypes": dtypes,
        "target": {"positive": pos, "negative": neg, "prevalence_pct": round(prev * 100, 4)},
        "month_counts": {int(k): int(v) for k, v in n_per_month.items()},
        "fraud_per_month": {int(k): float(v) for k, v in fraud_per_month.items()},
        "categorical_cardinality": cat_card,
        "categorical_top_values": {c: {str(k): int(v) for k, v in d.items()} for c, d in cat_top.items()},
        "missing_nan": {c: int(v) for c, v in missing.items()},
        "sentinel_neg1_pct": sentinel_pct,
        "top_abs_correlation_with_target": top_corr,
        "duplicate_pct_sampled": round(dup_pct_sample, 4),
        "skew_top": dict(top_skew),
        "numeric_unique_counts_first10": dict(list(nunique.items())[:10]),
    }


def main():
    summaries = {}
    for name, fname in FILES.items():
        summaries[name] = summarize(name, DATA / fname)

    # Cross-variant comparison
    print(f"\n{'='*72}\n  CROSS-VARIANT COMPARISON\n{'='*72}")
    print(f"  {'Variant':<12} {'rows':>10} {'cols':>5} {'pos':>10} {'prev%':>8}  feature_diffs")
    base_cols = set(pd.read_csv(DATA / FILES["Base"], nrows=1).columns)
    for n, s in summaries.items():
        cols_n = set(pd.read_csv(DATA / FILES[n], nrows=1).columns)
        diff = cols_n.symmetric_difference(base_cols)
        print(
            f"  {n:<12} {s['n_rows']:>10,} {s['n_cols']:>5} "
            f"{s['target']['positive']:>10,} {s['target']['prevalence_pct']:>7.3f}  {sorted(diff) if diff else '(same as Base)'}"
        )

    out_path = OUT / "exploration.json"
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
