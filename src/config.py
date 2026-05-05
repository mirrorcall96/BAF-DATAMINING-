"""Central configuration — single source of truth for paths, columns, hyperparams.

All other modules import from here so the pipeline stays consistent and any
change (paths, methods, hyperparameters) is made in exactly one place.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "outputs"
PREPROCESSED_DIR = OUTPUT_DIR / "preprocessed"
REPORT_DIR = OUTPUT_DIR / "report"
RESULTS_PATH = OUTPUT_DIR / "results.parquet"
FEATURE_CACHE_PATH = OUTPUT_DIR / "feature_cache.json"

for d in (OUTPUT_DIR, PREPROCESSED_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# BAF dataset variants
# ---------------------------------------------------------------------------
VARIANTS: dict[str, str] = {
    "Base":       "Base.csv",
    "VariantI":   "Variant I.csv",
    "VariantII":  "Variant II.csv",
    "VariantIII": "Variant III.csv",
    "VariantIV":  "Variant IV.csv",
    "VariantV":   "Variant V.csv",
}

# Target & temporal-split column
TARGET = "fraud_bool"
MONTH_COL = "month"
TRAIN_MONTHS = [0, 1, 2, 3, 4, 5]
TEST_MONTHS = [6, 7]

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
# Columns where -1 is a documented sentinel for missingness in BAF
SENTINEL_NEG1 = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "intended_balcon_amount",
    "session_length_in_minutes",
]

# Five categorical (object) columns — one-hot encoded by prep
CATEGORICAL = [
    "payment_type", "employment_status", "housing_status", "source", "device_os",
]

# Heavy-tailed numeric columns clipped via IQR (1.5 IQR rule from Lecture 2)
CLIP_COLUMNS = [
    "days_since_request", "intended_balcon_amount", "session_length_in_minutes",
    "bank_branch_count_8w", "velocity_6h", "velocity_24h", "velocity_4w",
    "zip_count_4w", "income", "name_email_similarity",
]

# Drop one of every pair of features with |Pearson rho| > this on training set
CORR_DROP_THRESHOLD = 0.95

# Used by SMOTE rounding (the one-hot prefixes get values rounded back to {0,1})
DUMMY_PREFIXES = CATEGORICAL

# ---------------------------------------------------------------------------
# Sweep axes
# ---------------------------------------------------------------------------
IMBALANCE_METHODS = ["none", "smote", "undersample"]
FS_METHODS = ["all", "mi", "ga"]
MODELS = ["DT", "SVM", "ANN", "NB", "RF", "XGBoost"]
COST_SENSITIVE_MODEL = "XGBoost"  # the one model the homework asks us to make cost-sensitive

# ---------------------------------------------------------------------------
# Feature-selection hyperparameters
# ---------------------------------------------------------------------------
MI_SUBSAMPLE = 100_000  # stratified subsample for fast mutual_info_classif

# Genetic Algorithm wrapper
GA_POP_SIZE = 30
GA_GENERATIONS = 15
GA_TOURNAMENT_K = 3
GA_P_CROSSOVER = 0.8
GA_ELITISM = 2
GA_SPARSITY_PENALTY = 0.001
GA_FITNESS_SUBSAMPLE = 30_000
GA_FITNESS_CV_FOLDS = 3

# ---------------------------------------------------------------------------
# Classifier hyperparameters
# ---------------------------------------------------------------------------
HP_DT = dict(max_depth=12, min_samples_leaf=50)
HP_SVM = dict(C=1.0, max_iter=5000, dual="auto")
HP_ANN = dict(
    hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
    learning_rate_init=1e-3, batch_size=512, max_iter=100,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
)
HP_RF = dict(n_estimators=200, max_depth=15, min_samples_leaf=20, n_jobs=-1)
HP_XGB = dict(
    n_estimators=500, max_depth=6, learning_rate=0.1,
    tree_method="hist", n_jobs=-1, eval_metric="auc", verbosity=0,
)
HP_LGBM = dict(
    n_estimators=1000, learning_rate=0.05, num_leaves=64,
    min_child_samples=20, n_jobs=-1, verbosity=-1,
)
HP_CATBOOST = dict(
    iterations=2000, depth=8, learning_rate=0.05,
    verbose=0,
)

NEEDS_SCALING = {"SVM", "ANN", "NB"}

# Models that can use GPU when available
GPU_CAPABLE = {"XGBoost", "XGBoost-tuned", "XGBoost-CS", "LightGBM", "LightGBM-bal", "CatBoost"}
