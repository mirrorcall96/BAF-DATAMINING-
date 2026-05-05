"""Step 3 — Feature selection.

Three methods (Lecture 3):
  - "all" : no selection (baseline)
  - "mi"  : Mutual Information filter, top-K by mutual_info_classif
  - "ga"  : Genetic Algorithm wrapper using LogisticRegression as fitness evaluator

GA implementation is custom (no DEAP dependency) and uses the operators named
in the lecture: tournament selection, uniform crossover, bit-flip mutation, elitism.
All hyperparameters are configured in `config.py`.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample as sk_resample

from .config import (
    GA_ELITISM, GA_FITNESS_CV_FOLDS, GA_FITNESS_SUBSAMPLE, GA_GENERATIONS,
    GA_P_CROSSOVER, GA_POP_SIZE, GA_SPARSITY_PENALTY, GA_TOURNAMENT_K,
    MI_SUBSAMPLE, RANDOM_STATE,
)

FsMethod = Literal["all", "mi", "ga", "rfe", "mrmr", "boruta", "permutation"]


# ---------------------------------------------------------------------------
# Mutual Information (regular filter)
# ---------------------------------------------------------------------------
def select_mi(
    X: pd.DataFrame, y: pd.Series, k: int | None = None,
) -> tuple[list[str], pd.Series]:
    """Top-k features by mutual_info_classif (computed on a stratified subsample
    if X is large). k defaults to ceil(n_features / 2)."""
    if k is None:
        k = max(1, X.shape[1] // 2)

    if len(X) > MI_SUBSAMPLE:
        X_s, y_s = sk_resample(
            X, y, n_samples=MI_SUBSAMPLE, replace=False,
            random_state=RANDOM_STATE, stratify=y,
        )
    else:
        X_s, y_s = X, y

    mi = mutual_info_classif(X_s, y_s, random_state=RANDOM_STATE, n_jobs=-1)
    scores = pd.Series(mi, index=X.columns, name="mi").sort_values(ascending=False)
    return scores.head(k).index.tolist(), scores


# ---------------------------------------------------------------------------
# Genetic Algorithm (evolutionary wrapper)
# ---------------------------------------------------------------------------
def _evaluate_individual(
    mask: np.ndarray, X_sub: np.ndarray, y_sub: np.ndarray,
    cv: StratifiedKFold, sparsity_penalty: float,
) -> float:
    """Fitness = mean CV ROC-AUC of LR on the masked features, minus a small
    sparsity penalty so equal-AUC ties prefer smaller subsets."""
    n_selected = int(mask.sum())
    if n_selected == 0:
        return -1.0
    cols = np.where(mask)[0]
    X_sel = X_sub[:, cols]

    aucs = []
    for tr, va in cv.split(X_sel, y_sub):
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X_sel[tr])
        Xv = scaler.transform(X_sel[va])
        clf = LogisticRegression(solver="lbfgs", max_iter=200, random_state=RANDOM_STATE)
        clf.fit(Xt, y_sub[tr])
        proba = clf.predict_proba(Xv)[:, 1]
        aucs.append(roc_auc_score(y_sub[va], proba))
    return float(np.mean(aucs)) - sparsity_penalty * n_selected


def _tournament(pop, fits, k: int, rng) -> np.ndarray:
    idx = rng.choice(len(pop), size=k, replace=False)
    return pop[int(idx[int(np.argmax([fits[i] for i in idx]))])].copy()


def _uniform_crossover(a, b, rng):
    mask = rng.random(len(a)) < 0.5
    c1, c2 = a.copy(), b.copy()
    c1[mask], c2[mask] = b[mask], a[mask]
    return c1, c2


def _mutate(ind, p_per_gene, rng):
    flips = rng.random(len(ind)) < p_per_gene
    out = ind.copy()
    out[flips] = 1 - out[flips]
    return out


def select_ga(X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> tuple[list[str], dict]:
    """Run the GA wrapper feature selection. Returns (selected_columns, history)."""
    rng = np.random.default_rng(RANDOM_STATE)
    feature_names = list(X.columns)
    n_feat = len(feature_names)

    if len(X) > GA_FITNESS_SUBSAMPLE:
        X_s, y_s = sk_resample(
            X, y, n_samples=GA_FITNESS_SUBSAMPLE, replace=False,
            random_state=RANDOM_STATE, stratify=y,
        )
    else:
        X_s, y_s = X, y
    X_arr = X_s.to_numpy(dtype=np.float32)
    y_arr = y_s.to_numpy(dtype=np.int8)
    cv = StratifiedKFold(n_splits=GA_FITNESS_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    population = [rng.integers(0, 2, size=n_feat).astype(np.int8) for _ in range(GA_POP_SIZE)]
    population[0] = np.ones(n_feat, dtype=np.int8)  # seed with all-features
    fitnesses = [_evaluate_individual(ind, X_arr, y_arr, cv, GA_SPARSITY_PENALTY)
                 for ind in population]

    history = {"best_fitness": [], "mean_fitness": [], "best_n_features": []}
    p_mutation = 1.0 / n_feat

    for gen in range(GA_GENERATIONS):
        order = np.argsort(fitnesses)[::-1]
        population = [population[i] for i in order]
        fitnesses = [fitnesses[i] for i in order]
        history["best_fitness"].append(fitnesses[0])
        history["mean_fitness"].append(float(np.mean(fitnesses)))
        history["best_n_features"].append(int(population[0].sum()))
        if verbose:
            print(f"  GA gen {gen+1:2d}/{GA_GENERATIONS}  "
                  f"best_fit={fitnesses[0]:.4f}  mean={np.mean(fitnesses):.4f}  "
                  f"best_n_feat={int(population[0].sum())}")

        new_pop = [ind.copy() for ind in population[:GA_ELITISM]]
        while len(new_pop) < GA_POP_SIZE:
            p1 = _tournament(population, fitnesses, GA_TOURNAMENT_K, rng)
            p2 = _tournament(population, fitnesses, GA_TOURNAMENT_K, rng)
            if rng.random() < GA_P_CROSSOVER:
                c1, c2 = _uniform_crossover(p1, p2, rng)
            else:
                c1, c2 = p1, p2
            new_pop.append(_mutate(c1, p_mutation, rng))
            if len(new_pop) < GA_POP_SIZE:
                new_pop.append(_mutate(c2, p_mutation, rng))

        new_fits = list(fitnesses[:GA_ELITISM])
        for ind in new_pop[GA_ELITISM:]:
            new_fits.append(_evaluate_individual(ind, X_arr, y_arr, cv, GA_SPARSITY_PENALTY))
        population, fitnesses = new_pop, new_fits

    best_idx = int(np.argmax(fitnesses))
    best_mask = population[best_idx]
    selected = [feature_names[i] for i, b in enumerate(best_mask) if b == 1]
    history["final_best_fitness"] = float(fitnesses[best_idx])
    history["final_n_selected"] = len(selected)
    return selected, history


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def select_features(
    X: pd.DataFrame, y: pd.Series, method: FsMethod, k: int | None = None,
) -> tuple[list[str], dict]:
    """Dispatcher. Returns (selected_columns, info_dict)."""
    if method == "all":
        return list(X.columns), {"method": "all", "n_selected": X.shape[1]}
    if method == "mi":
        cols, scores = select_mi(X, y, k=k)
        return cols, {"method": "mi", "k": len(cols), "top_scores": scores.head(15).to_dict()}
    if method == "ga":
        cols, history = select_ga(X, y)
        return cols, {"method": "ga", "n_selected": len(cols), "history": history}
    if method == "rfe":
        cols, ranking = select_rfe(X, y, k=k)
        return cols, {"method": "rfe", "n_selected": len(cols), "ranking": ranking}
    if method == "mrmr":
        cols, info = select_mrmr(X, y, k=k)
        return cols, {"method": "mrmr", "n_selected": len(cols), **info}
    if method == "boruta":
        cols, info = select_boruta(X, y)
        return cols, {"method": "boruta", "n_selected": len(cols), **info}
    if method == "permutation":
        cols, info = select_permutation(X, y, k=k)
        return cols, {"method": "permutation", "n_selected": len(cols), **info}
    raise ValueError(f"Unknown FS method: {method!r}")


# ---------------------------------------------------------------------------
# mRMR (Peng, Long & Ding 2005) — minimum redundancy, maximum relevance
# ---------------------------------------------------------------------------
def select_mrmr(
    X: pd.DataFrame, y: pd.Series, k: int | None = None,
    sample_size: int = 100_000,
) -> tuple[list[str], dict]:
    if k is None:
        k = max(1, X.shape[1] // 2)
    try:
        from mrmr import mrmr_classif
    except ImportError as e:
        raise RuntimeError("Install `mrmr-selection` to use mRMR feature selection") from e

    if len(X) > sample_size:
        X_s, y_s = sk_resample(
            X, y, n_samples=sample_size, replace=False,
            random_state=RANDOM_STATE, stratify=y,
        )
    else:
        X_s, y_s = X, y
    selected = mrmr_classif(X=X_s, y=y_s, K=k, show_progress=False)
    return list(selected), {"k": k}


# ---------------------------------------------------------------------------
# Boruta (Kursa & Rudnicki 2010) — RF + shadow features
# ---------------------------------------------------------------------------
def select_boruta(
    X: pd.DataFrame, y: pd.Series,
    sample_size: int = 100_000,
    n_estimators: int = 100,
) -> tuple[list[str], dict]:
    try:
        from boruta import BorutaPy
    except ImportError as e:
        raise RuntimeError("Install `boruta` to use Boruta feature selection") from e

    if len(X) > sample_size:
        X_s, y_s = sk_resample(
            X, y, n_samples=sample_size, replace=False,
            random_state=RANDOM_STATE, stratify=y,
        )
    else:
        X_s, y_s = X, y
    rf = RandomForestClassifier(
        n_estimators=n_estimators, n_jobs=-1, max_depth=10,
        class_weight="balanced", random_state=RANDOM_STATE,
    )
    boruta = BorutaPy(
        rf, n_estimators="auto", verbose=0, random_state=RANDOM_STATE, max_iter=50,
    )
    # BorutaPy expects numpy arrays
    boruta.fit(X_s.values, y_s.values)
    selected = [c for c, sup in zip(X.columns, boruta.support_) if sup]
    if not selected:
        # Fallback: tentative features (rank ≤ 2) so we never return zero
        selected = [c for c, r in zip(X.columns, boruta.ranking_) if r <= 2]
    if not selected:
        # Final fallback: keep top-k by ranking
        k = max(1, X.shape[1] // 2)
        ranks = list(zip(X.columns, boruta.ranking_))
        ranks.sort(key=lambda kv: kv[1])
        selected = [c for c, _ in ranks[:k]]
    ranking = {c: int(r) for c, r in zip(X.columns, boruta.ranking_)}
    return selected, {"ranking": ranking}


# ---------------------------------------------------------------------------
# Permutation importance (Fisher, Rudin & Dominici 2019) — model-faithful
# ---------------------------------------------------------------------------
def select_permutation(
    X: pd.DataFrame, y: pd.Series, k: int | None = None,
    sample_size: int = 50_000,
    n_repeats: int = 5,
) -> tuple[list[str], dict]:
    if k is None:
        k = max(1, X.shape[1] // 2)

    if len(X) > sample_size:
        X_s, y_s = sk_resample(
            X, y, n_samples=sample_size, replace=False,
            random_state=RANDOM_STATE, stratify=y,
        )
    else:
        X_s, y_s = X, y

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, n_jobs=-1,
        class_weight="balanced", random_state=RANDOM_STATE,
    )
    rf.fit(X_s, y_s)
    perm = permutation_importance(
        rf, X_s, y_s, n_repeats=n_repeats, random_state=RANDOM_STATE,
        n_jobs=-1, scoring="roc_auc",
    )
    importances = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
    selected = importances.head(k).index.tolist()
    return selected, {"importances_top": importances.head(15).to_dict()}


# ---------------------------------------------------------------------------
# Recursive Feature Elimination (wrapper, with LR base)
# ---------------------------------------------------------------------------
def select_rfe(
    X: pd.DataFrame, y: pd.Series, k: int | None = None,
    sample_size: int = 50_000,
) -> tuple[list[str], dict]:
    """RFE with logistic regression as the base estimator. We use a stratified
    subsample because RFE refits the estimator |F| - k times."""
    if k is None:
        k = max(1, X.shape[1] // 2)

    if len(X) > sample_size:
        X_s, y_s = sk_resample(
            X, y, n_samples=sample_size, replace=False,
            random_state=RANDOM_STATE, stratify=y,
        )
    else:
        X_s, y_s = X, y

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(solver="lbfgs", max_iter=300, random_state=RANDOM_STATE)),
    ])
    rfe = RFE(estimator=pipeline.named_steps["lr"], n_features_to_select=k, step=2)
    Xs_arr = StandardScaler().fit_transform(X_s)
    rfe.fit(Xs_arr, y_s)
    selected = [c for c, sup in zip(X.columns, rfe.support_) if sup]
    ranking = {c: int(r) for c, r in zip(X.columns, rfe.ranking_)}
    return selected, ranking


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    from .config import PREPROCESSED_DIR

    X = pd.read_parquet(PREPROCESSED_DIR / "Base" / "X_train.parquet")
    y = pd.read_parquet(PREPROCESSED_DIR / "Base" / "y_train.parquet").iloc[:, 0]

    print("MI feature selection ...")
    t0 = time.time()
    cols_mi, info_mi = select_features(X, y, "mi")
    print(f"  selected {len(cols_mi)}/{X.shape[1]} in {time.time()-t0:.1f}s")

    print("\nGA feature selection ...")
    t0 = time.time()
    cols_ga, info_ga = select_features(X, y, "ga")
    print(f"  selected {len(cols_ga)}/{X.shape[1]} in {time.time()-t0:.1f}s")
    print(f"  final fitness: {info_ga['history']['final_best_fitness']:.4f}")
