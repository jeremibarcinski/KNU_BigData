"""
Stage 4 (ASOS): modeling - predict day D+1 from the data of day D.

Input:   ASOS_03_Daegu_daily_features.csv   (output of A_ASOS_03_make_features.py)
Output:  ASOS_04_model_results.csv           (metrics table, one row per target / model / split)
         ASOS_04_modeling_report.txt         (text report: tuning, thresholds, importance, errors)
         ASOS_04_pr_curves_val.png           (precision-recall curves on val)

Targets: y_tropical_night, y_heatwave_advisory, y_heatwave_advisory_night (classification)
         y_tmax_feels (regression, optional part)
The alarm targets are not trained (almost no positive days).

Steps:
  1. load, split by the `split` column (no shuffling)
  2. baselines: persistence and climatology
  3. logistic regression (median imputer -> scaler -> model)
  4. tree models: HistGradientBoosting, LightGBM, XGBoost (they handle NaN, no imputer)
  5. tuning with expanding-window CV by year inside train (PR-AUC); class weighting
     (none / balanced) is a tuned option, because the threshold already handles the imbalance
     and balanced weights make the probabilities too high (worse Brier score)
  6. comparison on val
  7. decision threshold chosen on val (highest threshold with recall >= MIN_RECALL), then fixed
  8. final model refit on train + val (only if EVALUATE_TEST)
  9. test evaluated once (only if EVALUATE_TEST)
 10. permutation importance on val and error analysis

The test split stays closed while EVALUATE_TEST = False. Make every choice on val first,
then set EVALUATE_TEST = True and run the script ONE more time. Do not change anything after that.

Requirements:  pip install pandas numpy scikit-learn matplotlib lightgbm xgboost
               (macOS: brew install libomp, needed by lightgbm and xgboost)
Run:           python A_ASOS_04_modeling.py
"""

from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_curve, precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ---------- SETTINGS ----------
INPUT_CSV = "ASOS_03_Daegu_daily_features.csv"
RESULTS_CSV = "ASOS_04_model_results.csv"
REPORT_TXT = "ASOS_04_modeling_report.txt"
PR_PLOT = "ASOS_04_pr_curves_val.png"

EVALUATE_TEST = True         # final run: all choices were made on val with False; do not change anything now

TARGETS = ["y_tropical_night", "y_heatwave_advisory", "y_heatwave_advisory_night"]
REG_TARGET = "y_tmax_feels"
NOT_FEATURES = ["split", "check_tmax_air_next_day"]

# Persistence thresholds (same as the target definitions)
FEELS_ADVISORY = 33.0
TROPICAL_NIGHT_THRESHOLD = 25.0
ADVISORY_NIGHT_THRESHOLD = 26.0

CLIMATOLOGY_WINDOW = 7       # +- days around the day of year
CV_FIRST_VAL_YEAR = 2017     # expanding window: 2014-2016 -> 2017, 2014-2017 -> 2018, ... -> 2022
RANDOM_STATE = 42

# Small grids: only about 1368 training rows. "balanced" = weight positives by negatives / positives.
WEIGHTS = [None, "balanced"]
LOGREG_GRID = {"class_weight": WEIGHTS, "C": [0.01, 0.1, 1.0]}
HGB_GRID = {"class_weight": WEIGHTS, "learning_rate": [0.03, 0.1], "max_depth": [3, None],
            "min_samples_leaf": [20, 50]}
LGBM_GRID = {"class_weight": WEIGHTS, "learning_rate": [0.03, 0.1], "num_leaves": [7, 15],
             "min_child_samples": [20, 50]}
XGB_GRID = {"class_weight": WEIGHTS, "learning_rate": [0.03, 0.1], "max_depth": [2, 4],
            "min_child_weight": [1, 5]}
HGB_FIXED = {"max_iter": 300, "max_leaf_nodes": 15, "l2_regularization": 1.0}
LGBM_FIXED = {"n_estimators": 300, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
              "reg_lambda": 1.0, "verbose": -1, "n_jobs": 1}
XGB_FIXED = {"n_estimators": 300, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "n_jobs": 1}

# Threshold rule (README: "the best F1, or a minimum recall such as 0.8").
# Recall first: for a heat warning, a missed hot day is worse than a false alarm.
MIN_RECALL = 0.8

TOP_FEATURES = 10

report: list[str] = []
rows: list[dict] = []


def log(text: str = "") -> None:
    print(text)
    report.append(text)


# ---------- STEP 1 ----------
def load() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"], index_col="date").sort_index()
    y_cols = [c for c in df.columns if c.startswith("y_")]
    x_cols = [c for c in df.columns if c not in y_cols + NOT_FEATURES]
    log("1. DATA")
    log(f"   {INPUT_CSV}: {len(df)} rows, {len(x_cols)} features")
    log("   Rows per split: " + ", ".join(f"{s}: {(df['split'] == s).sum()}" for s in ["train", "val", "test"]))
    log(f"   Test evaluation: {'ON (final run)' if EVALUATE_TEST else 'OFF (test stays closed)'}")
    return df, x_cols


# ---------- STEP 2 ----------
def persistence(df: pd.DataFrame, target: str) -> pd.Series:
    """Tomorrow is like today: 0/1 prediction from the last known day / night."""
    hot = df["hot33_streak"] > 0
    warm25 = df["night_min_air_prev"] >= TROPICAL_NIGHT_THRESHOLD
    warm26 = df["night_min_air_prev"] >= ADVISORY_NIGHT_THRESHOLD
    pred = {"y_tropical_night": warm25,
            "y_heatwave_advisory": hot,
            "y_heatwave_advisory_night": hot & warm26}[target]
    return pred.astype(float)


def climatology(fit: pd.DataFrame, df: pd.DataFrame, target: str) -> pd.Series:
    """Share of positive days in `fit` for the same day of year (+- CLIMATOLOGY_WINDOW days)."""
    doy_fit = fit.index.dayofyear.to_numpy()
    y_fit = fit[target].to_numpy()
    table = {}
    for d in np.unique(df.index.dayofyear):
        near = np.abs(doy_fit - d) <= CLIMATOLOGY_WINDOW
        table[d] = y_fit[near].mean() if near.any() else y_fit.mean()
    return pd.Series([table[d] for d in df.index.dayofyear], index=df.index)


# ---------- STEPS 3-4 ----------
# Every factory gets the training target y (XGBoost needs it for scale_pos_weight).
def make_logreg(y, **params):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=5000, **params))


def make_hgb(y, **params):
    return HistGradientBoostingClassifier(random_state=RANDOM_STATE, **HGB_FIXED, **params)


def make_lgbm(y, **params):
    return LGBMClassifier(random_state=RANDOM_STATE, **LGBM_FIXED, **params)


def make_xgb(y, class_weight=None, **params):
    spw = (y == 0).sum() / max((y == 1).sum(), 1) if class_weight == "balanced" else 1.0
    return XGBClassifier(random_state=RANDOM_STATE, scale_pos_weight=spw, **XGB_FIXED, **params)


MODELS = [("logistic regression", make_logreg, LOGREG_GRID),
          ("hist gradient boosting", make_hgb, HGB_GRID),
          ("lightgbm", make_lgbm, LGBM_GRID),
          ("xgboost", make_xgb, XGB_GRID)]


# ---------- STEP 5 ----------
def year_folds(train: pd.DataFrame):
    years = train.index.year
    for y in range(CV_FIRST_VAL_YEAR, years.max() + 1):
        yield years < y, years == y


def tune(make, grid: dict, train: pd.DataFrame, x_cols: list[str], target: str) -> tuple[dict, float]:
    best, best_score = None, -np.inf
    for values in product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        scores = []
        for fit_mask, val_mask in year_folds(train):
            y_val = train.loc[val_mask, target]
            if y_val.sum() == 0:      # PR-AUC undefined for a year without positive days
                continue
            y_fit = train.loc[fit_mask, target]
            model = make(y_fit, **params).fit(train.loc[fit_mask, x_cols], y_fit)
            scores.append(average_precision_score(y_val, model.predict_proba(train.loc[val_mask, x_cols])[:, 1]))
        score = float(np.mean(scores))
        if score > best_score:
            best, best_score = params, score
    return best, best_score


# ---------- STEP 7 ----------
def choose_threshold(y: pd.Series, p: np.ndarray) -> float:
    """Highest threshold that still reaches recall >= MIN_RECALL (fewest false alarms at that recall)."""
    _, recall, thr = precision_recall_curve(y, p)
    return float(thr[recall[:-1] >= MIN_RECALL].max())


def evaluate(target: str, model: str, split: str, y: pd.Series, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    row = {
        "target": target, "model": model, "split": split,
        "positive_share_%": round(100 * y.mean(), 1),
        "PR_AUC": round(average_precision_score(y, p), 3),
        "precision": round(precision_score(y, pred, zero_division=0), 3),
        "recall": round(recall_score(y, pred, zero_division=0), 3),
        "F1": round(f1_score(y, pred, zero_division=0), 3),
        "Brier": round(brier_score_loss(y, np.clip(p, 0, 1)), 3),
        "threshold": round(thr, 3),
    }
    rows.append(row)
    return row


# ---------- STEP 10 ----------
def explain(model, X_val: pd.DataFrame, y_val: pd.Series, p_val: np.ndarray, thr: float) -> None:
    imp = permutation_importance(model, X_val, y_val, scoring="average_precision",
                                 n_repeats=10, random_state=RANDOM_STATE)
    order = np.argsort(imp.importances_mean)[::-1][:TOP_FEATURES]
    log(f"   Permutation importance on val (drop in PR-AUC), top {TOP_FEATURES}:")
    for i in order:
        log(f"       {X_val.columns[i]:32s} {imp.importances_mean[i]:+.3f} (+- {imp.importances_std[i]:.3f})")

    pred = p_val >= thr
    fn = y_val.index[(y_val == 1) & ~pred]
    fp = y_val.index[(y_val == 0) & pred]
    log(f"   Errors on val: {len(fn)} missed days (FN), {len(fp)} false alarms (FP)")
    log("       FN per month: " + str(pd.Series(fn.month).value_counts().sort_index().to_dict()))
    log("       FP per month: " + str(pd.Series(fp.month).value_counts().sort_index().to_dict()))


def classification(df: pd.DataFrame, x_cols: list[str]) -> None:
    train, val, test = (df[df["split"] == s] for s in ["train", "val", "test"])
    trainval = df[df["split"] != "test"]
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(5.5 * len(TARGETS), 4.8))

    for ax, target in zip(axes, TARGETS):
        log(f"\n=== {target} ===")
        log(f"   Positive share: train {100 * train[target].mean():.1f}%, val {100 * val[target].mean():.1f}%, "
            f"test {100 * test[target].mean():.1f}%")
        X_tr, y_tr, X_va, y_va = train[x_cols], train[target], val[x_cols], val[target]

        # baselines
        p_pers = persistence(val, target).to_numpy()
        evaluate(target, "persistence", "val", y_va, p_pers, 0.5)
        p_clim = climatology(train, val, target).to_numpy()
        thr_clim = choose_threshold(y_va, p_clim)
        evaluate(target, "climatology", "val", y_va, p_clim, thr_clim)

        # tuned models
        candidates = {}
        for name, make, grid in MODELS:
            params, cv = tune(make, grid, train, x_cols, target)
            model = make(y_tr, **params).fit(X_tr, y_tr)
            p_va = model.predict_proba(X_va)[:, 1]
            thr = choose_threshold(y_va, p_va)
            row = evaluate(target, name, "val", y_va, p_va, thr)
            log(f"   {name}: best params {params}, CV PR-AUC {cv:.3f}, val PR-AUC {row['PR_AUC']:.3f}, "
                f"threshold {thr:.3f}")
            candidates[name] = (make, params, model, p_va, thr, row["PR_AUC"])

            precision, recall, _ = precision_recall_curve(y_va, p_va)
            ax.plot(recall, precision, label=f"{name} (AP {row['PR_AUC']:.2f})")

        precision, recall, _ = precision_recall_curve(y_va, p_clim)
        ax.plot(recall, precision, "--", label="climatology")
        r_p = recall_score(y_va, p_pers.astype(int)), precision_score(y_va, p_pers.astype(int), zero_division=0)
        ax.plot(*r_p, "ko", label="persistence")
        ax.axhline(y_va.mean(), color="grey", lw=0.8, ls=":")
        ax.set(title=target, xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=8, loc="lower left")

        # best model = highest PR-AUC on val
        best_name = max(candidates, key=lambda k: candidates[k][5])
        make, params, model, p_va, thr, _ = candidates[best_name]
        log(f"   Best on val: {best_name}")
        explain(model, X_va, y_va, p_va, thr)

        if EVALUATE_TEST:
            final = make(trainval[target], **params).fit(trainval[x_cols], trainval[target])
            evaluate(target, f"best model ({best_name})", "test", test[target],
                     final.predict_proba(test[x_cols])[:, 1], thr)
            evaluate(target, "persistence", "test", test[target], persistence(test, target).to_numpy(), 0.5)
            evaluate(target, "climatology", "test", test[target],
                     climatology(trainval, test, target).to_numpy(), thr_clim)

    fig.tight_layout()
    fig.savefig(PR_PLOT, dpi=120)
    plt.close(fig)


# ---------- REGRESSION (optional) ----------
def regression(df: pd.DataFrame, x_cols: list[str]) -> None:
    log(f"\n=== {REG_TARGET} (regression) ===")
    train, val, test = (df[df["split"] == s] for s in ["train", "val", "test"])
    trainval = df[df["split"] != "test"]
    models = {
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
        "gradient boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE, learning_rate=0.05,
                                                           max_iter=300, max_leaf_nodes=15, min_samples_leaf=20),
    }

    def score(name: str, split: str, y: pd.Series, p) -> None:
        mae = mean_absolute_error(y, p)
        rmse = float(np.sqrt(mean_squared_error(y, p)))
        rows.append({"target": REG_TARGET, "model": name, "split": split, "MAE": round(mae, 3),
                     "RMSE": round(rmse, 3)})
        log(f"   {split:5s} {name:22s} MAE {mae:.2f}  RMSE {rmse:.2f}")

    splits = [("val", train, val)] + ([("test", trainval, test)] if EVALUATE_TEST else [])
    for split, fit, ev in splits:
        score("persistence", split, ev[REG_TARGET], ev["feels_like_max"])
        for name, model in models.items():
            score(name, split, ev[REG_TARGET], clone(model).fit(fit[x_cols], fit[REG_TARGET]).predict(ev[x_cols]))


def main() -> None:
    df, x_cols = load()
    log(f"\n2-7. BASELINES, MODELS, TUNING (expanding window by year), THRESHOLD ON VAL (recall >= {MIN_RECALL})")
    classification(df, x_cols)
    regression(df, x_cols)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_CSV, index=False)
    log("\nRESULTS")
    for line in results.fillna("").to_string(index=False).splitlines():
        log("   " + line)
    log("\nNotes: thresholds are chosen on val and kept fixed; persistence is a 0/1 rule (threshold 0.5).")
    log("Limitations: heat wave labels use D+2 (run definition); hourly data underestimate KMA extremes;")
    log("feels-like is computed, not downloaded; test set is small (304 days) and results are noisy.")

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\nSaved: {RESULTS_CSV}, {REPORT_TXT}, {PR_PLOT}")


if __name__ == "__main__":
    main()
