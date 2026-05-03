"""End-to-end pipeline runner — does everything from raw CSV to trained models.

Pipeline:
  1. Data preparation     (src/prep.py)        outputs/preprocessed/
  2. Imbalance handling   (src/imbalance.py)
  3. Feature selection    (src/featsel.py)
  4-5. Train classifiers  (src/models.py + src/train.py)
  6. Cost-sensitive XGB   (src/train.py)
  7. Evaluate             (src/evaluate.py)    outputs/results.parquet

Reports are intentionally NOT run by main.py — generate them on demand:
    python -m src.reports.markdown
    python -m src.reports.tex
    python -m src.reports.report_time_tex   # extended Base sweep only

Usage
-----
    python main.py                          # full pipeline, all 6 variants
    python main.py --variants Base          # only the Base variant
    python main.py --skip-prep              # skip Step 1 (already done)
    python main.py --skip-train             # skip Steps 2-7 (already done)
    python main.py --skip-cs                # skip cost-sensitive XGBoost
"""
from __future__ import annotations

import argparse
import sys
import time

from src.config import DATA_DIR, VARIANTS
from src.prep import main as run_prep
from src.train import run as run_train


def banner(title: str) -> None:
    print("\n" + "#" * 78)
    print(f"#  {title}")
    print("#" * 78)


def check_data_exists() -> None:
    missing = [fname for _, fname in VARIANTS.items() if not (DATA_DIR / fname).exists()]
    if missing:
        print("ERROR: missing data files in", DATA_DIR)
        for m in missing:
            print(f"   - {m}")
        print("\nDownload all 6 CSVs from:")
        print("   https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022")
        print(f"and place them in {DATA_DIR}/")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bank Account Fraud — end-to-end pipeline")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS.keys()),
                    help="variants to run (default: all 6)")
    ap.add_argument("--skip-prep", action="store_true", help="skip Step 1 (data prep)")
    ap.add_argument("--skip-train", action="store_true", help="skip Steps 2-7 (training)")
    ap.add_argument("--skip-cs", action="store_true", help="skip cost-sensitive XGBoost")
    args = ap.parse_args()

    t_total = time.time()
    check_data_exists()

    if not args.skip_prep:
        banner("Step 1 — Data preparation")
        run_prep()
    else:
        print("\n[skip] data preparation")

    if not args.skip_train:
        banner("Steps 2-7 — Training sweep (imbalance x feature-selection x models)")
        run_train(args.variants, skip_cs=args.skip_cs)
    else:
        print("\n[skip] training")

    elapsed = time.time() - t_total
    print(f"\nTotal pipeline runtime: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
