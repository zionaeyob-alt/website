"""AutoML training and evaluation with FLAML (Steps 4 and 5).

Usage (from the project root):
    python src/train_and_evaluate.py            # 180 s search per model (default)
    python src/train_and_evaluate.py --budget 60

Trains two models on the same stratified 80/20 split of data/cleaned_bank_data.csv:
    primary    realistic pre-call model, `duration` dropped
    benchmark  `duration` kept, to measure the leakage effect

Outputs:
    models/automl_{primary,benchmark}.pkl, models/model_metadata.json
    figures/05_confusion_matrices.png ... figures/08_leakage_comparison.png
    report/automl_evaluation_summary.md
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))  # so pickles reference src.learners, not __main__

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flaml import AutoML
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.learners import ScaledLogisticRegression

warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
warnings.filterwarnings("ignore", category=FutureWarning)

CLEAN_PATH = PROJECT_ROOT / "data" / "cleaned_bank_data.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
SUMMARY_PATH = PROJECT_ROOT / "report" / "automl_evaluation_summary.md"

SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 5
ESTIMATORS = ["lgbm", "xgboost", "rf", "lr"]
LEARNER_NAMES = {"lgbm": "LightGBM", "xgboost": "XGBoost", "rf": "Random Forest",
                 "lr": "Logistic Regression"}
LEAKY = "duration"
DPI = 300

# Reference palette (see EDA script): primary = slot 1 blue, benchmark = slot 2 orange.
PRIMARY = "#2a78d6"
BENCH = "#eb6834"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"])


# --------------------------------------------------------------------------- helpers
def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False, "legend.fontsize": 9,
        "text.color": INK,
    })


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES_DIR / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figures/{name}")


def md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def fmt_config(cfg: dict) -> str:
    out = []
    for k, v in cfg.items():
        v = float(v) if isinstance(v, (np.floating, float)) else v
        out.append(f"`{k}` = {v:.4g}" if isinstance(v, float) else f"`{k}` = {v}")
    return ", ".join(out)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_PATH)
    cat_cols = df.select_dtypes(exclude="number").columns
    # Categories fixed on the full dataset so train and test share identical levels.
    return df.astype({c: "category" for c in cat_cols})


# --------------------------------------------------------------------------- modelling
def train_automl(X: pd.DataFrame, y: pd.Series, name: str, budget: int) -> AutoML:
    log_path = MODELS_DIR / f"flaml_{name}.log"
    automl = AutoML()
    automl.add_learner("lr", ScaledLogisticRegression)
    automl.fit(
        X, y,
        task="classification", metric="roc_auc",
        eval_method="cv", n_splits=N_FOLDS, split_type="stratified",
        estimator_list=ESTIMATORS, time_budget=budget, seed=SEED,
        log_file_name=str(log_path), log_type="all", verbose=0,
    )
    return automl


def oof_probabilities(automl: AutoML, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Out-of-fold probabilities for the best config, refit on each of 5 stratified folds.

    These come only from training data, so a threshold tuned on them never sees the test set.
    """
    est_cls = type(automl.best_model_for_estimator(automl.best_estimator))
    config = {k: v for k, v in automl.best_config.items() if k != "FLAML_sample_size"}
    Xp = automl.preprocess(X)
    oof = np.zeros(len(y))
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, va in folds.split(Xp, y):
        est = est_cls(task="binary", **config)
        est.fit(Xp.iloc[tr], y.iloc[tr])
        oof[va] = est.predict_proba(Xp.iloc[va])[:, 1]
    return oof


def best_f1_threshold(y: pd.Series, proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def metrics_at(y: pd.Series, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y, pred),
        "Balanced Acc.": balanced_accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "ROC-AUC": roc_auc_score(y, proba),
        "PR-AUC": average_precision_score(y, proba),
    }


def search_trajectory(log_path: Path) -> pd.DataFrame:
    records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    rows = [r for r in records if "learner" in r and "validation_loss" in r]
    traj = pd.DataFrame({
        "trial": [r["record_id"] for r in rows],
        "learner": [r["learner"] for r in rows],
        "wall_clock_s": [r["wall_clock_time"] for r in rows],
        "cv_auc": [1 - r["validation_loss"] for r in rows],
        "config": [r["config"] for r in rows],
    })
    traj["best_so_far"] = traj["cv_auc"].cummax()
    return traj


def permutation_importance_auc(model, X: pd.DataFrame, y: pd.Series, repeats: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    rows = []
    for col in X.columns:
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[col] = Xp[col].to_numpy()[rng.permutation(len(Xp))] if Xp[col].dtype != "category" \
                else pd.Categorical(Xp[col].to_numpy()[rng.permutation(len(Xp))],
                                    categories=X[col].cat.categories)
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        rows.append({"feature": col, "mean": np.mean(drops), "std": np.std(drops)})
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)


def gains_at(y: pd.Series, proba: np.ndarray, frac: float) -> tuple[float, float]:
    """Precision and share of all subscribers captured when calling the top `frac` of clients."""
    k = int(round(len(y) * frac))
    top = np.argsort(-proba)[:k]
    hits = y.to_numpy()[top].sum()
    return hits / k, hits / y.sum()


# --------------------------------------------------------------------------- figures
def plot_confusion(res: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.5))
    fig.subplots_adjust(left=0.1, right=0.84, top=0.86, bottom=0.08, wspace=0.42, hspace=0.5)
    for r, key in enumerate(["primary", "benchmark"]):
        for c, (tlabel, thr) in enumerate([("default", 0.5), ("tuned", res[key]["threshold"])]):
            ax = axes[r, c]
            pred = (res[key]["test_proba"] >= thr).astype(int)
            cm = confusion_matrix(res[key]["y_test"], pred)
            row_pct = cm / cm.sum(axis=1, keepdims=True)
            ax.imshow(row_pct, cmap=SEQ_BLUE, vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i, j]:,}\n{row_pct[i, j]:.1%} of row", ha="center", va="center",
                            fontsize=11, color="white" if row_pct[i, j] > 0.55 else INK)
            ax.set_xticks([0, 1], ["Predicted no", "Predicted yes"])
            ax.set_yticks([0, 1], ["Actual no", "Actual yes"])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            p, rc = precision_score(res[key]["y_test"], pred), recall_score(res[key]["y_test"], pred)
            name = "Primary (no duration)" if key == "primary" else "Benchmark (with duration)"
            ax.set_title(f"{name}\n{tlabel.capitalize()} threshold = {thr:.3f}\n"
                         f"Precision {p:.1%} · Recall {rc:.1%}", fontsize=10.5)
    sm = plt.cm.ScalarMappable(cmap=SEQ_BLUE, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm, cax=fig.add_axes([0.88, 0.25, 0.018, 0.45]),
                        format=matplotlib.ticker.PercentFormatter(1))
    cbar.set_label("Share of the actual class (row %)")
    fig.suptitle("Confusion matrices on the 20% test set: default (0.5) vs tuned threshold",
                 x=0.02, ha="left", fontsize=14, fontweight="bold", y=0.97)
    fig.text(0.02, 0.0, "Tuned threshold maximises F1 on out-of-fold predictions from the training "
             "set only; the test set was not used to choose it.", fontsize=8, color=MUTED)
    save(fig, "05_confusion_matrices.png")


def plot_curves(res: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))
    prevalence = res["primary"]["y_test"].mean()
    for key, color, label in [("primary", PRIMARY, "Primary (no duration)"),
                              ("benchmark", BENCH, "Benchmark (with duration)")]:
        y, p, thr = res[key]["y_test"], res[key]["test_proba"], res[key]["threshold"]
        fpr, tpr, _ = roc_curve(y, p)
        ax1.plot(fpr, tpr, color=color, linewidth=2, label=f"{label}: AUC {roc_auc_score(y, p):.3f}")
        prec, rec, _ = precision_recall_curve(y, p)
        ax2.plot(rec, prec, color=color, linewidth=2,
                 label=f"{label}: AP {average_precision_score(y, p):.3f}")
        pred = p >= thr
        op_fpr = (pred & (y == 0)).sum() / (y == 0).sum()
        op_tpr = (pred & (y == 1)).sum() / (y == 1).sum()
        op_prec = precision_score(y, pred)
        for ax, (xv, yv) in [(ax1, (op_fpr, op_tpr)), (ax2, (op_tpr, op_prec))]:
            ax.scatter([xv], [yv], s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=5)
    ax1.plot([0, 1], [0, 1], color=MUTED, linewidth=1, label="No-skill baseline (AUC 0.5)")
    ax2.axhline(prevalence, color=MUTED, linewidth=1, label=f"No-skill baseline (= positive rate {prevalence:.3f})")
    ax1.scatter([], [], s=60, color=INK_2, label="Tuned-threshold operating point")
    ax2.scatter([], [], s=60, color=INK_2, label="Tuned-threshold operating point")
    ax1.set(xlabel="False positive rate (non-subscribers called)",
            ylabel="True positive rate (subscribers found)", xlim=(0, 1), ylim=(0, 1.01))
    ax2.set(xlabel="Recall (share of subscribers found)", ylabel="Precision (share of calls that convert)",
            xlim=(0, 1), ylim=(0, 1.01))
    ax1.set_title("ROC curve (test set)")
    ax2.set_title("Precision–recall curve (test set)")
    ax1.legend(loc="lower right")
    ax2.legend(loc="upper right")
    fig.suptitle("Ranking quality on the 20% holdout", x=0.0, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=3)
    save(fig, "06_roc_pr_curves.png")


def plot_importance(imp: pd.DataFrame, model_name: str, top: int = 15) -> None:
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(d["feature"], d["mean"], xerr=d["std"], color=PRIMARY, height=0.7, zorder=2,
            error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2},
            label="Mean drop in test ROC-AUC (±1 SD, 5 shuffles)")
    for i, (v, sd) in enumerate(zip(d["mean"], d["std"])):
        label = f"{v:.3f}" if abs(v) >= 0.0005 else "≈0"
        ax.text(max(v, 0) + sd + 0.0008, i, label, va="center", fontsize=8, color=INK_2)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in ROC-AUC when the feature is randomly shuffled")
    ax.set_title(f"Top {top} features of the primary model ({model_name}, no duration)\n"
                 "Permutation importance on the test set")
    ax.legend(loc="lower right")
    fig.text(0.0, -0.02, "Correlated features (e.g. the macro indicators) share importance: shuffling "
             "one leaves the others to carry the signal, so each looks smaller than the group.",
             fontsize=8, color=MUTED)
    save(fig, "07_feature_importance.png")


def plot_leakage(res: dict) -> None:
    names = ["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "Balanced Acc."]
    p = [res["primary"]["tuned"][m] for m in names]
    b = [res["benchmark"]["tuned"][m] for m in names]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.bar(x - w / 2 - 0.01, p, w, color=PRIMARY, label="Primary: realistic, no duration", zorder=2)
    ax.bar(x + w / 2 + 0.01, b, w, color=BENCH, label="Benchmark: with duration (leaky)", zorder=2)
    for i in range(len(names)):
        ax.text(x[i] - w / 2 - 0.01, p[i] + 0.012, f"{p[i]:.2f}", ha="center", fontsize=8.5, color=INK_2)
        ax.text(x[i] + w / 2 + 0.01, b[i] + 0.012, f"{b[i]:.2f}", ha="center", fontsize=8.5, color=INK_2)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score on the 20% test set")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=2, bbox_to_anchor=(1, 1.02))
    headline = ("Including call duration inflates every metric" if all(bv > pv for pv, bv in zip(p, b))
                else "Including call duration inflates most metrics")
    ax.set_title(f"{headline}\n"
                 "Threshold-based metrics use each model's tuned threshold", pad=24)
    fig.text(0.0, -0.02, "Duration is only known after the call ends, so the benchmark cannot be used "
             "to decide whom to call. It shows how much leakage would overstate performance.",
             fontsize=8, color=MUTED)
    save(fig, "08_leakage_comparison.png")


# --------------------------------------------------------------------------- summary
def write_summary(res: dict, split: dict, budget: int) -> str:
    P, B = res["primary"], res["benchmark"]
    traj = P["trajectory"]
    best_per = {k: 1 - v for k, v in P["automl"].best_loss_per_estimator.items() if v is not None}
    learner_stats = pd.DataFrame({
        "trials": traj.groupby("learner").size().reindex(list(best_per)).fillna(0).astype(int),
        "best_cv_auc": pd.Series(best_per),
    }).sort_values("best_cv_auc", ascending=False)
    gaps = learner_stats["best_cv_auc"].iloc[0] - learner_stats["best_cv_auc"]
    explored = [i for i in learner_stats.index if learner_stats.loc[i, "trials"] >= 5]
    sparse = [i for i in learner_stats.index if learner_stats.loc[i, "trials"] < 5]
    leader = learner_stats.index[0]
    others = [i for i in explored if i != leader]
    parts = []
    if others:
        parts.append(f"Among learners searched at least 5 times, {LEARNER_NAMES[leader]} led "
                     + ", ".join(f"{LEARNER_NAMES[i]} by {gaps[i]:.4f}" for i in others)
                     + " CV ROC-AUC.")
        if max(gaps[i] for i in others) < 0.02:
            parts.append("These gaps are small: the main signal (macro indicators, month, contact "
                         "type, contact history) is captured by every well-searched learner.")
    if sparse:
        parts.append("**Under-explored:** " + ", ".join(
            f"{LEARNER_NAMES[i]} ({learner_stats.loc[i, 'trials']} trials, best {learner_stats.loc[i, 'best_cv_auc']:.4f})"
            for i in sparse)
            + ". Their scores are lower bounds from a few configurations, not a fair verdict on the "
              "algorithm; a longer budget would be needed to compare them properly.")
    search_commentary = " ".join(parts)
    learner_tbl = pd.DataFrame({
        "Learner": [LEARNER_NAMES[i] for i in learner_stats.index],
        "Trials": learner_stats["trials"].values,
        "Best 5-fold CV ROC-AUC": [f"{v:.4f}" for v in learner_stats["best_cv_auc"]],
        "Best config": [fmt_config(P["automl"].best_config_per_estimator[i]) for i in learner_stats.index],
    })
    improvements = traj[traj["cv_auc"] >= traj["best_so_far"]].drop_duplicates("best_so_far")
    imp_tbl = pd.DataFrame({
        "Trial": improvements["trial"].values,
        "Time (s)": [f"{v:.1f}" for v in improvements["wall_clock_s"]],
        "Learner": [LEARNER_NAMES[i] for i in improvements["learner"]],
        "CV ROC-AUC": [f"{v:.4f}" for v in improvements["cv_auc"]],
    })

    rows = []
    for label, r in [("Primary (no duration)", P), ("Benchmark (with duration)", B)]:
        for tname, key in [("0.5", "default"), (f"{r['threshold']:.3f} (tuned)", "tuned")]:
            rows.append({"Model": label, "Threshold": tname,
                         **{k: f"{v:.3f}" for k, v in r[key].items()}})
    metrics_tbl = pd.DataFrame(rows)

    def cm_counts(r, thr):
        tn, fp, fn, tp = confusion_matrix(r["y_test"], (r["test_proba"] >= thr).astype(int)).ravel()
        return tn, fp, fn, tp

    tn0, fp0, fn0, tp0 = cm_counts(P, 0.5)
    tn1, fp1, fn1, tp1 = cm_counts(P, P["threshold"])
    n_test, n_pos = split["n_test"], split["n_test_pos"]
    base = n_pos / n_test
    g10, g20, g30 = (gains_at(P["y_test"], P["test_proba"], f) for f in (0.1, 0.2, 0.3))
    top_feats = P["importance"].head(8)
    d_auc = B["tuned"]["ROC-AUC"] - P["tuned"]["ROC-AUC"]
    d_pr = B["tuned"]["PR-AUC"] - P["tuned"]["PR-AUC"]
    d_f1 = B["tuned"]["F1"] - P["tuned"]["F1"]
    dur_rank = int(B["importance"].index[B["importance"]["feature"] == LEAKY][0]) + 1
    dur_drop = float(B["importance"].loc[B["importance"]["feature"] == LEAKY, "mean"].iloc[0])

    md = f"""# AutoML Training and Evaluation

Script: `src/train_and_evaluate.py` · Framework: FLAML {__import__('flaml').__version__} ·
Data: `data/cleaned_bank_data.csv` ({split['n_total']:,} rows)

## 1. Setup

| Item | Setting |
|---|---|
| Split | Stratified 80/20, `random_state={SEED}`: {split['n_train']:,} train ({split['train_rate']:.2%} positive), {n_test:,} test ({base:.2%} positive) |
| Primary model | {len(P['features'])} features, `duration` **dropped** (realistic pre-call targeting) |
| Benchmark model | {len(B['features'])} features, `duration` **kept** (leakage demonstration) |
| Task / metric | Binary classification, optimise ROC-AUC |
| Validation | {N_FOLDS}-fold stratified cross-validation inside the training set |
| Search space | LightGBM, XGBoost, Random Forest, Logistic Regression (one-hot + scaled, custom learner) |
| Budget | {budget} s per model (identical for both, so the comparison is fair) |
| Threshold tuning | F1-maximising threshold chosen on out-of-fold predictions of the best config ({N_FOLDS} stratified folds of the training set). The test set is used only for final evaluation |
| Saved artifacts | `models/automl_primary.pkl`, `models/automl_benchmark.pkl`, `models/model_metadata.json` |

The categorical features are passed to FLAML as pandas `category` columns: LightGBM and XGBoost use
native categorical splits, and the logistic regression one-hot encodes them. FLAML's search is
time-based, so a re-run can pick slightly different hyperparameters; the split and seeds are fixed.

## 2. Primary model: search results

**Chosen model family: {LEARNER_NAMES[P['automl'].best_estimator]}**, best 5-fold CV ROC-AUC
**{1 - P['automl'].best_loss:.4f}**, found after {P['automl'].time_to_find_best_model:.0f} s
(trial {P['automl'].best_iteration} of {len(traj)}).

Best hyperparameters: {fmt_config(P['automl'].best_config)}.

### Search trajectory across algorithms

{md_table(learner_tbl)}

FLAML's cost-aware search starts every learner from a cheap configuration and gives more trials to
learners that improve fastest per second of compute, so trial counts differ by learner
({len(traj)} trials in total) and also between runs, because the budget is wall-clock time.
{search_commentary}

Each time the overall best improved:

{md_table(imp_tbl)}

## 3. Test-set evaluation (20% holdout, {n_test:,} clients, {n_pos:,} subscribers)

{md_table(metrics_tbl)}

- **CV vs test:** the primary model's test ROC-AUC ({P['tuned']['ROC-AUC']:.4f}) is close to its CV
  estimate ({1 - P['automl'].best_loss:.4f}), so the search did not overfit the validation folds.
- **ROC-AUC {P['tuned']['ROC-AUC']:.3f}:** a randomly chosen subscriber is ranked above a randomly chosen
  non-subscriber {P['tuned']['ROC-AUC']:.0%} of the time.
- **PR-AUC {P['tuned']['PR-AUC']:.3f}** against a no-skill baseline of {base:.3f}, i.e.
  {P['tuned']['PR-AUC'] / base:.1f}× better than random targeting across all recall levels.
- **Accuracy is not informative here:** the default threshold scores {P['default']['Accuracy']:.1%}
  accuracy, barely above the {1 - base:.1%} obtained by predicting "no" for everyone.

### Default (0.5) vs tuned threshold ({P['threshold']:.3f})

With an 11% positive rate, predicted probabilities rarely exceed 0.5, so the default threshold
calls very few clients:

| Primary model | Clients flagged | Subscribers found (TP) | Missed subscribers (FN) | Wasted calls (FP) | Precision | Recall |
|---|---|---|---|---|---|---|
| Threshold 0.5 | {tp0 + fp0:,} | {tp0:,} | {fn0:,} | {fp0:,} | {P['default']['Precision']:.1%} | {P['default']['Recall']:.1%} |
| Threshold {P['threshold']:.3f} | {tp1 + fp1:,} | {tp1:,} | {fn1:,} | {fp1:,} | {P['tuned']['Precision']:.1%} | {P['tuned']['Recall']:.1%} |

Lowering the threshold to {P['threshold']:.3f} finds **{tp1 - tp0:+,} more subscribers** at the cost of
**{fp1 - fp0:+,} extra calls** to non-subscribers, raising F1 from {P['default']['F1']:.3f} to
{P['tuned']['F1']:.3f} and balanced accuracy from {P['default']['Balanced Acc.']:.3f} to
{P['tuned']['Balanced Acc.']:.3f}.

## 4. Interpretation in a banking context

**The two errors have very different costs.**

- A **false positive** is a call to a client who will not subscribe. Its cost is agent time and
  telephony (a few minutes per call), plus a small risk of annoying the client.
- A **false negative** is a client who would have subscribed but is never called. The cost is the
  lost margin on a term deposit over its whole life, typically far larger than the cost of one call.

So when call capacity allows, the bank should **favour recall over precision**, i.e. a threshold
below the F1-optimal one. The F1 threshold is a neutral starting point; the operating threshold
should be set from the actual cost per call, the expected margin per deposit, and the number of
calls the team can make. For example, if a successful call nets 10× what an unsuccessful call costs,
calling has positive expected value whenever p × 10 > (1 − p), i.e. above p = 1/11 ≈ 0.09.

**Ranking view (how campaigns are actually run).** Sorting test clients by predicted probability:

| Call the top… | Conversion rate of those calls | Share of all subscribers reached | Lift vs random |
|---|---|---|---|
| 10% | {g10[0]:.1%} | {g10[1]:.1%} | {g10[0] / base:.1f}× |
| 20% | {g20[0]:.1%} | {g20[1]:.1%} | {g20[0] / base:.1f}× |
| 30% | {g30[0]:.1%} | {g30[1]:.1%} | {g30[0] / base:.1f}× |

Random calling converts {base:.1%}. Calling only the top-scored 20% of clients reaches
{g20[1]:.0%} of all subscribers with {g20[0] / base:.1f}× the conversion rate, i.e. roughly 80% fewer
calls for {g20[1]:.0%} of the business.

**What drives the primary model** (permutation importance, Figure 7):
{", ".join(f"`{r.feature}` ({r.mean:.3f})" for r in top_feats.itertuples())}.
This agrees with the EDA: macro conditions, contact history, timing and channel matter most;
personal demographics matter less.

## 5. Impact of `duration` (data leakage)

| Metric (tuned threshold) | Primary (no duration) | Benchmark (with duration) | Difference |
|---|---|---|---|
| ROC-AUC | {P['tuned']['ROC-AUC']:.3f} | {B['tuned']['ROC-AUC']:.3f} | {d_auc:+.3f} |
| PR-AUC | {P['tuned']['PR-AUC']:.3f} | {B['tuned']['PR-AUC']:.3f} | {d_pr:+.3f} |
| F1 | {P['tuned']['F1']:.3f} | {B['tuned']['F1']:.3f} | {d_f1:+.3f} |
| Precision | {P['tuned']['Precision']:.3f} | {B['tuned']['Precision']:.3f} | {B['tuned']['Precision'] - P['tuned']['Precision']:+.3f} |
| Recall | {P['tuned']['Recall']:.3f} | {B['tuned']['Recall']:.3f} | {B['tuned']['Recall'] - P['tuned']['Recall']:+.3f} |

- The benchmark ({LEARNER_NAMES[B['automl'].best_estimator]}, CV ROC-AUC {1 - B['automl'].best_loss:.4f})
  looks far better: PR-AUC rises by {d_pr:+.3f} ({B['tuned']['PR-AUC'] / P['tuned']['PR-AUC']:.1f}×) and F1 by {d_f1:+.3f}.
- `duration` is the benchmark's **#{dur_rank} most important feature**: shuffling it alone lowers test
  ROC-AUC by {dur_drop:.3f}.
- This gain is not usable. Call duration is only known once the call has ended, when the outcome is
  already known, and a long call is partly a *result* of the client being interested. A model that
  depends on it cannot rank clients before calling them.
- The benchmark therefore shows how much leakage would **overstate** performance. Reporting it as the
  model's accuracy would mislead stakeholders. The primary model's results are the honest estimate.

## 6. Limitations

- Random 80/20 split: the data is ordered in time, and macro indicators act as time-period proxies,
  so a time-based split would give a more conservative estimate for future campaigns.
- Hyperparameters were tuned under a {budget}-second budget; a longer search may add a little.
- Probabilities were not calibrated. If expected-value thresholds are used operationally, check
  calibration first (e.g. with a reliability curve).
"""
    SUMMARY_PATH.write_text(md, encoding="utf-8")
    print(f"saved {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    return md


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=180, help="FLAML time budget per model (s)")
    args = parser.parse_args()

    set_style()
    for d in (FIGURES_DIR, MODELS_DIR, SUMMARY_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)

    df = load_data()
    y = df.pop("y")
    X_train_all, X_test_all, y_train, y_test = train_test_split(
        df, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    split = {"n_total": len(df), "n_train": len(y_train), "n_test": len(y_test),
             "train_rate": y_train.mean(), "n_test_pos": int(y_test.sum())}
    print(f"Train {len(y_train):,} ({y_train.mean():.2%} positive) | "
          f"Test {len(y_test):,} ({y_test.mean():.2%} positive)")

    res = {}
    configs = {"primary": [c for c in df.columns if c != LEAKY], "benchmark": list(df.columns)}
    for name, features in configs.items():
        print(f"\n=== {name}: {len(features)} features, budget {args.budget}s ===")
        X_tr, X_te = X_train_all[features], X_test_all[features]
        t0 = time.time()
        automl = train_automl(X_tr, y_train, name, args.budget)
        print(f"best: {automl.best_estimator}  CV ROC-AUC {1 - automl.best_loss:.4f}  "
              f"({time.time() - t0:.0f}s)")
        oof = oof_probabilities(automl, X_tr, y_train)
        threshold = best_f1_threshold(y_train, oof)
        proba = automl.predict_proba(X_te)[:, 1]
        res[name] = {
            "automl": automl, "features": features, "threshold": threshold,
            "y_test": y_test, "test_proba": proba,
            "oof_auc": roc_auc_score(y_train, oof),
            "default": metrics_at(y_test, proba, 0.5),
            "tuned": metrics_at(y_test, proba, threshold),
            "trajectory": search_trajectory(MODELS_DIR / f"flaml_{name}.log"),
            "importance": permutation_importance_auc(automl, X_te, y_test),
        }
        joblib.dump(automl, MODELS_DIR / f"automl_{name}.pkl")
        print(f"tuned threshold {threshold:.3f} (OOF ROC-AUC {res[name]['oof_auc']:.4f}); "
              f"saved models/automl_{name}.pkl")

    table = pd.DataFrame(
        [{"model": n, "threshold": t if t == "0.5" else f"{res[n]['threshold']:.3f}",
          **res[n]["default" if t == "0.5" else "tuned"]}
         for n in res for t in ("0.5", "tuned")]).set_index(["model", "threshold"])
    print("\nTest-set comparison:\n" + table.round(4).to_string())

    metadata = {
        n: {"best_estimator": r["automl"].best_estimator,
            "best_config": {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                            for k, v in r["automl"].best_config.items()},
            "cv_roc_auc": 1 - r["automl"].best_loss, "threshold": r["threshold"],
            "features": r["features"],
            "test_metrics_tuned": {k: float(v) for k, v in r["tuned"].items()},
            "test_metrics_default": {k: float(v) for k, v in r["default"].items()}}
        for n, r in res.items()
    }
    metadata["split"] = {"test_size": TEST_SIZE, "random_state": SEED, "stratified": True,
                         "budget_s": args.budget}
    (MODELS_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("saved models/model_metadata.json")

    plot_confusion(res)
    plot_curves(res)
    plot_importance(res["primary"]["importance"], LEARNER_NAMES[res["primary"]["automl"].best_estimator])
    plot_leakage(res)
    print("\n" + write_summary(res, split, args.budget))


if __name__ == "__main__":
    main()
