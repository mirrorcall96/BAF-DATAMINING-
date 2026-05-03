# Bank Account Fraud — Imbalanced Classification

## How to run

```bash
# 1. Set up Python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Mac only — XGBoost needs OpenMP
brew install libomp

# 2. Get the data — place all 6 CSVs in dataset/
#    https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

# 3. Run everything
python main.py
```

That's it. `main.py` runs all 8 steps: data prep → resampling × feature selection × 6 classifiers + cost-sensitive XGBoost on all 6 BAF variants → reports. Results stream live to the terminal and are saved to `outputs/`.

## Selective runs

```bash
python main.py --variants Base          # one variant only
python main.py --skip-prep              # skip data prep (already done)
python main.py --skip-train             # only build reports
python main.py --no-reports             # skip report generation
```

`main.py` is **resumable** — already-trained (variant, imbalance, feature-selection, model) cells in `outputs/results.parquet` are skipped automatically.

## Extended Base sweep (288 cells, deeper research dive)

```bash
python -m src.extended_base                  # 6 imbalance x 4 FS x 12 models
python -m src.reports.report_time_tex        # builds outputs/report/report_time.tex
```

Adds three more imbalance methods (ADASYN, SMOTE+ENN, SMOTE+Tomek), one more
feature-selection method (RFE), four more classifier configs (DT-shallow,
ANN-deep, RF-200, XGBoost-tuned, AdaBoost, HistGB), and three new metrics
(PR-AUC, TPR@5%FPR, threshold-tuned F1⁺). Per-cell artefacts go to
`outputs/runs/<run_id>.json`.
