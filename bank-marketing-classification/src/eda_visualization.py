"""Exploratory data analysis of the cleaned Bank Marketing dataset.

Usage (from the project root):
    python src/eda_visualization.py

Reads data/cleaned_bank_data.csv (with `duration`, which is fine for EDA but
must be dropped for modelling) and writes:
    figures/01_target_distribution.png
    figures/02_correlation_heatmap.png
    figures/03_categorical_conversion_rates.png
    figures/04_duration_and_economics.png
    report/eda_summary.md
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = PROJECT_ROOT / "data" / "cleaned_bank_data.csv"
FIGURES_DIR = PROJECT_ROOT / "figures"
SUMMARY_PATH = PROJECT_ROOT / "report" / "eda_summary.md"
DPI = 300

NUMERIC = ["age", "duration", "campaign", "pdays", "previous", "previously_contacted",
           "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
MACRO = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "day_of_week", "poutcome"]
LABELS = {
    "age": "Age", "duration": "Call duration", "campaign": "Contacts this campaign",
    "pdays": "Days since last contact", "previous": "Prior contacts",
    "previously_contacted": "Previously contacted", "emp.var.rate": "Employment var. rate",
    "cons.price.idx": "Consumer price index", "cons.conf.idx": "Consumer confidence",
    "euribor3m": "Euribor 3-month", "nr.employed": "Number employed", "y": "Subscribed (y)",
}

# Palette (validated reference palette): the positive class takes categorical slot 1 (blue);
# the negative class is a recessive neutral so "yes" carries the emphasis.
YES = "#2a78d6"
NO = "#a9a79f"
BAR = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
# Diverging blue <-> red with a neutral grey midpoint (no hue at zero).
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_grey_red", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#9e2a2a"])


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.labelsize": 10, "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 9,
        "text.color": INK,
    })


def save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.relative_to(PROJECT_ROOT)}")
    return path


def pct(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}%}"


def md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1))))


def rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.groupby(col, observed=True)["y"].agg(rate="mean", n="size")


def overall_line(ax: plt.Axes, rate: float, horizontal_bars: bool = True) -> None:
    """Draw the overall subscription rate as a solid hairline reference."""
    if horizontal_bars:
        ax.axvline(rate, color=INK_2, linewidth=1, zorder=3)
    else:
        ax.axhline(rate, color=INK_2, linewidth=1, zorder=3)


# --------------------------------------------------------------------------- plots
def plot_target(df: pd.DataFrame) -> None:
    counts = df["y"].value_counts().sort_index()
    shares = counts / counts.sum()
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    labels = ["No (0)", "Yes (1)"]
    bars = ax.barh(labels, counts.values, color=[NO, YES], height=0.55, zorder=2)
    for bar, n, s in zip(bars, counts.values, shares.values):
        ax.text(bar.get_width() + counts.max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{n:,}  ({pct(s)})", va="center", ha="left", fontsize=10, color=INK)
    ax.set_xlim(0, counts.max() * 1.22)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("Number of clients")
    ax.set_ylabel("Subscribed to term deposit")
    ax.set_title(f"Target is imbalanced: only {pct(shares[1])} of clients subscribed "
                 f"(≈{counts[0] / counts[1]:.1f} : 1)")
    ax.legend(handles=[Patch(color=NO, label=f"No: {counts[0]:,}"),
                       Patch(color=YES, label=f"Yes: {counts[1]:,}")],
              loc="lower right", title="Class", title_fontsize=9)
    fig.text(0.0, -0.04, f"n = {len(df):,} clients after removing duplicates. "
             "Source: UCI Bank Marketing (bank-additional-full).", fontsize=8, color=MUTED)
    save(fig, "01_target_distribution.png")


def plot_correlations(df: pd.DataFrame) -> pd.DataFrame:
    cols = NUMERIC + ["y"]
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    sns.heatmap(corr, mask=mask, cmap=DIVERGING, vmin=-1, vmax=1, center=0, square=True,
                annot=True, fmt=".2f", annot_kws={"fontsize": 7.5},
                linewidths=1.5, linecolor=SURFACE,
                xticklabels=[LABELS[c] for c in cols], yticklabels=[LABELS[c] for c in cols],
                cbar_kws={"shrink": 0.7, "label": "Pearson correlation (r)"}, ax=ax)
    ax.grid(False)
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    # Outline the target row so it reads as the row that matters.
    n = len(cols)
    ax.add_patch(plt.Rectangle((0, n - 1), n, 1, fill=False, edgecolor=INK, linewidth=1.2, clip_on=False))
    ax.set_title("Correlation of numeric features with each other and the target\n"
                 "Macro indicators are strongly inter-correlated; target row outlined")
    fig.text(0.0, -0.1, "`duration` is shown for exploration only: it is known after the call "
             "and leaks the outcome. y is 0/1, so its row is the point-biserial correlation.",
             fontsize=8, color=MUTED)
    save(fig, "02_correlation_heatmap.png")
    return corr


def _rate_bars(ax: plt.Axes, rates: pd.DataFrame, title: str, overall: float,
               order: list | None = None, rename: dict | None = None) -> None:
    rates = rates.loc[order] if order else rates.sort_values("rate")
    labels = [rename.get(i, i) if rename else str(i) for i in rates.index]
    labels = [f"{lab}  (n={n:,})" for lab, n in zip(labels, rates["n"])]
    ax.barh(labels, rates["rate"], color=BAR, height=0.7, zorder=2)
    for i, r in enumerate(rates["rate"]):
        ax.text(r + 0.008, i, pct(r), va="center", ha="left", fontsize=8, color=INK_2, zorder=4,
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
    overall_line(ax, overall)
    ax.set_xlim(0, max(rates["rate"].max() * 1.22, overall * 1.5))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Subscription rate")
    ax.set_title(title, fontsize=11)


def plot_categorical(df: pd.DataFrame) -> None:
    overall = df["y"].mean()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), gridspec_kw={"height_ratios": [1.35, 1]})
    _rate_bars(axes[0, 0], rate_by(df, "job"), "By job", overall)
    _rate_bars(axes[0, 1], rate_by(df, "education"), "By education", overall)
    _rate_bars(axes[1, 0], rate_by(df, "poutcome"), "By outcome of previous campaign", overall,
               order=["nonexistent", "failure", "success"],
               rename={"nonexistent": "no previous campaign"})
    _rate_bars(axes[1, 1], rate_by(df, "previously_contacted"), "By previous contact (new feature)",
               overall, order=[0, 1], rename={0: "not contacted before", 1: "contacted before"})
    handles = [Patch(color=BAR, label="Subscription rate for the group"),
               plt.Line2D([], [], color=INK_2, linewidth=1, label=f"Overall rate ({pct(overall)})")]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    fig.suptitle("Subscription rate varies most with prior campaign outcome and job",
                 x=0.0, ha="left", fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.5, w_pad=3)
    fig.text(0.0, -0.01, "Groups with small n (e.g. education = illiterate, n=18) have wide "
             "uncertainty. Sorted by rate except where the order is meaningful.",
             fontsize=8, color=MUTED)
    save(fig, "03_categorical_conversion_rates.png")


def plot_duration_economics(df: pd.DataFrame) -> dict:
    overall = df["y"].mean()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    out = {}

    # (a) duration distribution by class, clipped at the 99th percentile for readability
    ax = axes[0, 0]
    cap = df["duration"].quantile(0.99)
    bins = np.linspace(0, cap, 50)
    for val, color, label in [(0, NO, "No"), (1, YES, "Yes")]:
        d = df.loc[df["y"] == val, "duration"].clip(upper=cap) / 60
        ax.hist(d, bins=bins / 60, density=True, color=color, alpha=0.55 if val == 0 else 0.6,
                label=f"{label} (median {d.median():.1f} min)", zorder=2, edgecolor=SURFACE, linewidth=0.3)
        ax.axvline(d.median(), color=color if val else INK_2, linewidth=1.5, zorder=3)
    ax.set_xlabel(f"Call duration (minutes; last bin holds all calls ≥ {cap / 60:.0f} min, the 99th pct.)")
    ax.set_ylabel("Density")
    ax.legend(title="Subscribed", title_fontsize=9)
    ax.set_title("(a) Subscribers had much longer calls", fontsize=11)

    # (b) rate by duration band
    ax = axes[0, 1]
    edges = [-1, 60, 120, 180, 300, 600, 1000, df["duration"].max()]
    names = ["≤1", "1–2", "2–3", "3–5", "5–10", "10–17", ">17"]
    dur = rate_by(df.assign(band=pd.cut(df["duration"], edges, labels=names)), "band")
    ax.bar(dur.index.astype(str), dur["rate"], color=BAR, width=0.7, zorder=2)
    for i, (r, n) in enumerate(zip(dur["rate"], dur["n"])):
        ax.text(i, r + 0.012, pct(r), ha="center", fontsize=8, color=INK_2)
    overall_line(ax, overall, horizontal_bars=False)
    ax.set_ylim(0, dur["rate"].max() * 1.15)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Call duration (minutes)")
    ax.set_ylabel("Subscription rate")
    ax.set_title("(b) Rate rises steadily with call length (leaky: known only after the call)", fontsize=11)
    out["duration"] = dur

    # (c) rate by euribor3m quintile
    ax = axes[1, 0]
    q = pd.qcut(df["euribor3m"], 5)
    eur = rate_by(df.assign(band=q), "band")
    eur_labels = [f"{iv.left:.2f}–{iv.right:.2f}" for iv in eur.index]
    ax.bar(eur_labels, eur["rate"], color=BAR, width=0.7, zorder=2)
    for i, (r, n) in enumerate(zip(eur["rate"], eur["n"])):
        ax.text(i, r + 0.006, f"{pct(r)}\nn={n:,}", ha="center", fontsize=8, color=INK_2)
    overall_line(ax, overall, horizontal_bars=False)
    ax.set_ylim(0, eur["rate"].max() * 1.25)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Euribor 3-month rate (%), quintile bands")
    ax.set_ylabel("Subscription rate")
    ax.set_title("(c) Low interest rates coincide with far higher subscription", fontsize=11)
    out["euribor"] = eur.set_axis(eur_labels)

    # (d) rate by employment variation rate
    ax = axes[1, 1]
    emp = rate_by(df, "emp.var.rate").sort_index()
    x = [f"{v:+.1f}" for v in emp.index]
    ax.bar(x, emp["rate"], color=BAR, width=0.7, zorder=2)
    for i, (r, n) in enumerate(zip(emp["rate"], emp["n"])):
        ax.text(i, r + 0.01, f"n={n:,}", ha="center", fontsize=7, color=MUTED, rotation=90, va="bottom")
    overall_line(ax, overall, horizontal_bars=False)
    ax.set_ylim(0, emp["rate"].max() * 1.35)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Employment variation rate (quarterly; each observed value, not to scale)")
    ax.set_ylabel("Subscription rate")
    ax.set_title("(d) Negative employment growth (recession) → higher subscription", fontsize=11)
    out["emp"] = emp

    handles = [Patch(color=NO, alpha=0.55, label="Not subscribed (a)"),
               Patch(color=YES, alpha=0.6, label="Subscribed (a)"),
               Patch(color=BAR, label="Subscription rate (b–d)"),
               plt.Line2D([], [], color=INK_2, linewidth=1, label=f"Overall rate ({pct(overall)})")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.965), ncol=4)
    fig.suptitle("Call duration and macroeconomic conditions vs. subscription",
                 x=0.0, ha="left", fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=3, w_pad=3)
    fig.text(0.0, -0.01, "Panel (d): employment variation of -0.2 has only n=10 clients. Macro indicators "
             "vary by contact date, so (c) and (d) reflect time periods (2008–2010).", fontsize=8, color=MUTED)
    save(fig, "04_duration_and_economics.png")
    return out


# --------------------------------------------------------------------------- write-up
def write_summary(df: pd.DataFrame, corr: pd.DataFrame, econ: dict) -> str:
    n = len(df)
    counts = df["y"].value_counts().sort_index()
    pos = counts[1] / n
    ratio = counts[0] / counts[1]

    # Univariate strength: numeric -> r and AUC; categorical -> Cramér's V and rate spread
    num_rows = []
    for c in NUMERIC:
        auc = roc_auc_score(df["y"], df[c])
        num_rows.append({"feature": c, "r": corr.loc[c, "y"], "auc": max(auc, 1 - auc),
                         "direction": "higher → more yes" if auc >= 0.5 else "higher → less yes"})
    num_df = pd.DataFrame(num_rows).sort_values("auc", ascending=False)

    cat_rows = []
    for c in CATEGORICAL:
        r = rate_by(df, c)
        r_big = r[r["n"] >= 100]  # ignore tiny groups when quoting extremes
        cat_rows.append({"feature": c, "v": cramers_v(df[c], df["y"]),
                         "lowest": f"{r_big['rate'].idxmin()} ({pct(r_big['rate'].min())})",
                         "highest": f"{r_big['rate'].idxmax()} ({pct(r_big['rate'].max())})"})
    cat_df = pd.DataFrame(cat_rows).sort_values("v", ascending=False)

    stats = df[NUMERIC].describe().T[["mean", "std", "min", "50%", "max"]]
    by_class = df.groupby("y")[NUMERIC].median().T.rename(columns={0: "median_no", 1: "median_yes"})
    stats = stats.join(by_class).reset_index().rename(columns={"index": "feature", "50%": "median"})
    for col in ["mean", "std", "min", "median", "max", "median_no", "median_yes"]:
        stats[col] = stats[col].map(lambda v: f"{v:,.2f}".rstrip("0").rstrip("."))

    macro_corr = corr.loc[MACRO, MACRO]
    pairs = [(a, b, macro_corr.loc[a, b]) for i, a in enumerate(MACRO) for b in MACRO[i + 1:]]
    top_pairs = sorted(pairs, key=lambda t: -abs(t[2]))[:4]

    job, edu, pout, prev = (rate_by(df, c) for c in ["job", "education", "poutcome", "previously_contacted"])
    month, contact = rate_by(df, "month"), rate_by(df, "contact")
    age = rate_by(df.assign(b=pd.cut(df["age"], [0, 24, 34, 44, 54, 64, 120],
                                      labels=["≤24", "25–34", "35–44", "45–54", "55–64", "65+"])), "b")
    camp = rate_by(df.assign(b=pd.cut(df["campaign"], [0, 1, 2, 3, 5, 10, 100],
                                       labels=["1", "2", "3", "4–5", "6–10", ">10"])), "b")
    dflt = rate_by(df, "default")
    eur, emp, dur = econ["euribor"], econ["emp"], econ["duration"]
    med_dur = df.groupby("y")["duration"].median()
    med_eur = df.groupby("y")["euribor3m"].median()

    num_tbl = num_df.assign(r=num_df["r"].map(lambda v: f"{v:+.3f}"),
                            auc=num_df["auc"].map(lambda v: f"{v:.3f}"))
    num_tbl.columns = ["Feature", "Pearson r with y", "Univariate ROC-AUC", "Direction"]
    cat_tbl = cat_df.assign(v=cat_df["v"].map(lambda v: f"{v:.3f}"))
    cat_tbl.columns = ["Feature", "Cramér's V", "Lowest rate (n ≥ 100)", "Highest rate (n ≥ 100)"]

    def rate_list(r: pd.DataFrame, keys) -> str:
        return ", ".join(f"{k} {pct(r.loc[k, 'rate'])}" for k in keys)

    md = f"""# Exploratory Data Analysis

Dataset: `data/cleaned_bank_data.csv` ({n:,} clients × {df.shape[1]} columns), the output of
`src/data_preprocessing.py`. `duration` is included here for exploration only; it is recorded
after the call ends, so it leaks the outcome and is excluded from the realistic model.

Figures (300 DPI) are in `figures/`:

| Figure | Content |
|---|---|
| `01_target_distribution.png` | Class balance of `y` (counts and percentages) |
| `02_correlation_heatmap.png` | Pearson correlations among numeric features and `y` |
| `03_categorical_conversion_rates.png` | Subscription rate by job, education, previous outcome, previous contact |
| `04_duration_and_economics.png` | Call duration vs. outcome; subscription rate vs. Euribor and employment variation |

## 1. Summary statistics

{md_table(stats)}

Observations:

- **Clients:** median age {df['age'].median():.0f} (range {df['age'].min()}–{df['age'].max()}).
  Subscribers and non-subscribers have almost the same median age; the age effect is
  non-linear (see section 4).
- **Campaign effort:** median {df['campaign'].median():.0f} contacts per client this campaign, with a
  long tail up to {df['campaign'].max()}. Only {pct(df['previously_contacted'].mean())} of clients had any
  prior contact (`previously_contacted`), so `pdays` and `previous` are mostly constant.
- **Duration:** median call {med_dur[1] / 60:.1f} min for subscribers vs {med_dur[0] / 60:.1f} min for
  non-subscribers ({med_dur[1] / med_dur[0]:.1f}× longer).
- **Macro context:** `euribor3m` is bimodal (clusters near 1% and near 4.9%), and its
  median is {med_eur[1]:.2f}% for subscribers vs {med_eur[0]:.2f}% for non-subscribers. These indicators
  vary by contact date, not by client, so they record *when* a client was called.

## 2. Key correlations (Figure 2)

**With the target.** The strongest linear associations with `y` are
{", ".join(f"`{r.feature}` (r = {r.r:+.2f})" for r in num_df.reindex(num_df['r'].abs().sort_values(ascending=False).index).head(5).itertuples())}.
Every macro indicator except consumer confidence is negatively correlated with subscription:
clients said yes more often when rates, employment and prices were low.

**Among features (multicollinearity).** The macro indicators are close to redundant:
{"; ".join(f"`{a}` ↔ `{b}` r = {v:+.2f}" for a, b, v in top_pairs)}.
`previous` and `previously_contacted` are also strongly linked (r = {corr.loc['previous', 'previously_contacted']:+.2f})
by construction, as are `pdays` and `previous` (r = {corr.loc['pdays', 'previous']:+.2f}). This matters for linear or logistic models (unstable coefficients; consider
keeping one or two indicators, or regularising) and for interpreting feature importances
(importance is split across correlated columns). Tree ensembles, as used by FLAML, are robust to it.

## 3. Class imbalance (Figure 1)

Only **{counts[1]:,} of {n:,} clients ({pct(pos)}) subscribed**, a ratio of about **{ratio:.1f} : 1**.

**Statistical impact**

- **Accuracy is misleading:** predicting "no" for everyone scores **{pct(1 - pos)} accuracy** while finding
  zero subscribers. Use ROC-AUC, **PR-AUC** (its no-skill baseline equals the positive rate, {pos:.3f}),
  F1, recall and precision for the positive class.
- **Learning bias:** unweighted models minimise overall error by under-predicting the minority class,
  and default probability outputs centre near the base rate. Remedies: class weights
  (`class_weight="balanced"` / `scale_pos_weight ≈ {ratio:.1f}`), tuning the decision threshold on
  validation data instead of using 0.5, and checking probability calibration.
- **Evaluation stability:** a test set of 20% holds only about {int(counts[1] * 0.2):,} positives, so
  metric estimates for the positive class are noisy. Use **stratified** train/test splits and
  stratified k-fold cross-validation. Resampling (SMOTE, undersampling), if used, must be applied only
  inside the training folds.

**Operational impact**

- Calling at random converts {pct(pos)} of contacts: about **{1 / pos:.1f} calls per subscription**.
  A model is valuable if it ranks clients so the top-scored segment converts well above that
  base rate (measure with lift / cumulative gains, precision@k).
- The costs of the two errors are unequal: a false positive costs one unnecessary call, while a false
  negative loses a deposit. The decision threshold should come from that cost trade-off and from
  call-centre capacity, not from the default 0.5.

## 4. What predicts subscription

### Univariate strength: numeric features

ROC-AUC of each feature used alone as a score (0.5 = no signal, symmetric in direction).

{md_table(num_tbl)}

### Univariate strength: categorical features

Cramér's V (0 = independent of `y`, 1 = perfectly associated). Extremes ignore groups with n < 100.

{md_table(cat_tbl)}

### Campaign attributes (most predictive group)

- **`duration` is the single strongest feature** (AUC {num_df.set_index('feature').loc['duration', 'auc']:.2f}):
  subscription goes from {pct(dur['rate'].iloc[0])} for calls of ≤1 min to {pct(dur['rate'].iloc[-1])} for calls
  over 17 min (Figure 4b). Because it is known only after the call, it is a *consequence* of the
  outcome, not a predictor for targeting, and is **excluded from the realistic model**. It can be
  used as a benchmark upper bound.
- **Previous campaign outcome** (`poutcome`, V = {cat_df.set_index('feature').loc['poutcome', 'v']:.2f}) is the
  strongest pre-call signal: {rate_list(pout, ['success', 'failure', 'nonexistent'])} (Figure 3).
  Any prior contact (`previously_contacted`) triples the rate: {pct(prev.loc[1, 'rate'])} vs
  {pct(prev.loc[0, 'rate'])}. Only {prev.loc[1, 'n']:,} clients fall in this group, so it is a high-value but small segment.
- **Contact month** (V = {cat_df.set_index('feature').loc['month', 'v']:.2f}): {rate_list(month, ['mar', 'dec', 'sep', 'oct'])}
  versus {pct(month.loc['may', 'rate'])} in May, which accounts for {pct(month.loc['may', 'n'] / n)} of all calls. The
  high-rate months are low-volume, so this may reflect campaign scheduling and targeting rather than
  true seasonality.
- **Contact channel:** cellular {pct(contact.loc['cellular', 'rate'])} vs telephone {pct(contact.loc['telephone', 'rate'])}.
- **Contact frequency:** the rate falls as contacts within this campaign rise:
  {", ".join(f"{k} call{'s' if k != '1' else ''} {pct(v)}" for k, v in camp['rate'].items())}. Repeated
  calls have diminishing returns.

### Macroeconomic factors

- `nr.employed`, `euribor3m` and `emp.var.rate` have univariate AUCs of
  {num_df.set_index('feature').loc['nr.employed', 'auc']:.2f}, {num_df.set_index('feature').loc['euribor3m', 'auc']:.2f} and
  {num_df.set_index('feature').loc['emp.var.rate', 'auc']:.2f}, making them the strongest legitimate numeric predictors.
- In the lowest Euribor quintile ({eur.index[0]}%) **{pct(eur['rate'].iloc[0])}** subscribed, versus
  {pct(eur['rate'].iloc[2])}–{pct(eur['rate'].iloc[-1])} in the top three quintiles (Figure 4c).
- Quarters with negative employment variation (e.g. {emp.index[0]:+.1f}: {pct(emp['rate'].iloc[0])}) convert at
  several times the rate of growth quarters (+1.1: {pct(emp.loc[1.1, 'rate'])}, +1.4: {pct(emp.loc[1.4, 'rate'])}) (Figure 4d).
- Interpretation: these indicators change with the contact date, not the client, so they act as
  time-period effects. A plausible explanation (not testable from this data alone) is that the
  low-rate, recessionary period coincided with smaller, more targeted campaigns and with safe deposits
  becoming more attractive to clients. Either way, they predict well within
  this 2008–2010 window but will **not transfer automatically to different economic regimes**, so the
  model should be monitored and retrained when conditions change. Because the data is ordered by date,
  a time-based validation split would test this directly.

### Client demographics

- **Job** (V = {cat_df.set_index('feature').loc['job', 'v']:.2f}): {rate_list(job, ['student', 'retired'])} are the most
  responsive groups; {rate_list(job, ['blue-collar', 'services'])} are the least (Figure 3).
- **Age** is U-shaped, which is why its linear correlation is near zero (r = {corr.loc['age', 'y']:+.2f}):
  {", ".join(f"{k}: {pct(v)}" for k, v in age['rate'].items())}. Tree models or age bands capture this.
- **Credit default status:** {pct(dflt.loc['no', 'rate'])} when `default = no` vs {pct(dflt.loc['unknown', 'rate'])} when
  `unknown`. The missingness itself is informative, which supports keeping `unknown` as a category.
- **Education** has a modest effect (V = {cat_df.set_index('feature').loc['education', 'v']:.2f}): university degree
  {pct(edu.loc['university.degree', 'rate'])} vs basic.9y {pct(edu.loc['basic.9y', 'rate'])}.
- **Weak or no signal:** `housing` (V = {cat_df.set_index('feature').loc['housing', 'v']:.3f}), `loan`
  (V = {cat_df.set_index('feature').loc['loan', 'v']:.3f}), `day_of_week` (V = {cat_df.set_index('feature').loc['day_of_week', 'v']:.3f})
  and `marital` (V = {cat_df.set_index('feature').loc['marital', 'v']:.3f}). These are candidates for removal
  if a simpler model is wanted.

## 5. Implications for modelling

1. **Drop `duration`** for the realistic model (`src.data.load_clean()` does this by default); report
   a with-duration model only as a benchmark.
2. **Expected strongest predictors:** macro indicators (`nr.employed`, `euribor3m`, `emp.var.rate`),
   `poutcome` / `previously_contacted` / `pdays`, `month`, `contact`, `job`, `age`, `default`.
3. **Handle imbalance:** stratified splits and CV, optimise ROC-AUC or PR-AUC (not accuracy), use
   class weights, and tune the decision threshold.
4. **Multicollinearity:** fine for tree ensembles; for linear baselines regularise or keep a subset of
   the macro indicators.
5. **Non-linearity:** age (U-shape), campaign (diminishing returns) and the bimodal macro variables
   favour tree-based learners, which is what FLAML searches over (LightGBM, XGBoost, random forest).
"""
    SUMMARY_PATH.write_text(md, encoding="utf-8")
    print(f"saved {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    return md


def main() -> None:
    set_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CLEAN_PATH)
    assert "duration" in df.columns, "duration must be present for EDA"
    assert set(df["y"].unique()) == {0, 1}, "run src/data_preprocessing.py first"
    print(f"Loaded {CLEAN_PATH.relative_to(PROJECT_ROOT)}: {df.shape[0]:,} rows × {df.shape[1]} cols")

    plot_target(df)
    corr = plot_correlations(df)
    plot_categorical(df)
    econ = plot_duration_economics(df)
    print("\n" + write_summary(df, corr, econ))


if __name__ == "__main__":
    main()
