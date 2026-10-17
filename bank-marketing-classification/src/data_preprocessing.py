"""Load, verify and clean the UCI Bank Marketing dataset (bank-additional-full.csv).

Usage (from the project root):
    python src/data_preprocessing.py

Outputs:
    data/cleaned_bank_data.csv                  cleaned dataset (duration kept for EDA)
    report/data_preprocessing_summary.md        markdown summary of every check and step
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "bank-additional-full.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "cleaned_bank_data.csv"
SUMMARY_PATH = PROJECT_ROOT / "report" / "data_preprocessing_summary.md"

EXPECTED_SHAPE = (41188, 21)
EXPECTED_COLUMNS = [
    "age", "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
    "previous", "poutcome", "emp.var.rate", "cons.price.idx",
    "cons.conf.idx", "euribor3m", "nr.employed", "y",
]
NUMERIC_COLUMNS = [
    "age", "duration", "campaign", "pdays", "previous", "emp.var.rate",
    "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]
CATEGORICAL_COLUMNS = [c for c in EXPECTED_COLUMNS if c not in NUMERIC_COLUMNS + ["y"]]
OUTLIER_CHECK_COLUMNS = ["age", "duration", "campaign", "previous"]

PDAYS_SENTINEL = 999
# Known only after the call ends, so it leaks the target. Kept in the cleaned file for EDA;
# drop it before training (see src.data.load_clean).
LEAKY_COLUMNS = ["duration"]


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Render a DataFrame as a GitHub markdown table (avoids the optional tabulate dependency)."""
    if index:
        df = df.reset_index()
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    md: list[str] = ["# Data Preprocessing Summary", "",
                     f"Source: `data/raw/{RAW_PATH.name}` (semicolon-delimited)", ""]

    # ------------------------------------------------------------------ 1. Load & verify
    section("1. LOAD & VERIFY")
    df = pd.read_csv(RAW_PATH, sep=";")

    shape_ok = df.shape == EXPECTED_SHAPE
    columns_ok = list(df.columns) == EXPECTED_COLUMNS
    numeric_ok = all(pd.api.types.is_numeric_dtype(df[c]) for c in NUMERIC_COLUMNS)
    categorical_ok = all(not pd.api.types.is_numeric_dtype(df[c]) for c in CATEGORICAL_COLUMNS + ["y"])
    n_missing = int(df.isna().sum().sum())
    target_values = sorted(df["y"].unique())
    target_ok = target_values == ["no", "yes"]

    print(f"Shape: {df.shape} (expected {EXPECTED_SHAPE}) -> {'OK' if shape_ok else 'MISMATCH'}")
    print(f"Columns match expected names and order -> {'OK' if columns_ok else 'MISMATCH'}")
    if not columns_ok:
        print(f"  missing: {set(EXPECTED_COLUMNS) - set(df.columns)}; "
              f"unexpected: {set(df.columns) - set(EXPECTED_COLUMNS)}")
    print(f"\nData types:\n{df.dtypes.to_string()}")
    print(f"\nNull values (NaN): {n_missing}")
    print(f"Target values: {target_values}")

    print("\nNumeric summary statistics:")
    num_stats = df[NUMERIC_COLUMNS].describe().T
    print(num_stats.round(3).to_string())

    print("\nCategorical summary:")
    cat_stats = df[CATEGORICAL_COLUMNS + ["y"]].describe().T
    print(cat_stats.to_string())

    if not (shape_ok and columns_ok and numeric_ok and categorical_ok and target_ok):
        raise SystemExit("Verification failed; inspect the raw file before cleaning.")

    def check(ok: bool) -> str:
        return "Pass" if ok else "**Fail**"

    md += [
        "## 1. Verification checks", "",
        "| Check | Expected | Observed | Result |", "|---|---|---|---|",
        f"| Dimensions | {EXPECTED_SHAPE[0]:,} rows × {EXPECTED_SHAPE[1]} cols "
        f"| {df.shape[0]:,} rows × {df.shape[1]} cols | {check(shape_ok)} |",
        f"| Column names & order | 21 documented attributes | {len(df.columns)} columns, matching "
        f"| {check(columns_ok)} |",
        f"| Numeric dtypes | 10 numeric features | "
        f"{sum(pd.api.types.is_numeric_dtype(df[c]) for c in NUMERIC_COLUMNS)} numeric "
        f"(5 int64, 5 float64) | {check(numeric_ok)} |",
        f"| Categorical dtypes | 10 categorical features + target | all string | {check(categorical_ok)} |",
        f"| Null values (NaN) | none documented | {n_missing} | {check(n_missing == 0)} |",
        f"| Target values | `no`, `yes` | {', '.join(f'`{v}`' for v in target_values)} | {check(target_ok)} |",
        "",
        "Missing values are not stored as NaN: the documentation states they are coded as the "
        "string `unknown` in categorical columns (see section 3).", "",
        "### Numeric summary statistics (raw)", "",
        md_table(num_stats.round(2).rename_axis("feature")), "",
    ]

    # ------------------------------------------------------------------ 2. Duplicates
    section("2. DUPLICATE ROWS")
    n_before = len(df)
    dup_mask = df.duplicated(keep="first")
    n_dups = int(dup_mask.sum())
    dup_target = df.loc[dup_mask, "y"].value_counts().to_dict()
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    print(f"Exact duplicate rows found: {n_dups} (target among duplicates: {dup_target})")
    print(f"Rows: {n_before:,} -> {len(df):,}")

    md += [
        "## 2. Duplicate rows", "",
        f"- Found **{n_dups} exact duplicate rows** (all 21 columns identical); "
        f"target among the duplicates: {', '.join(f'{k}={v}' for k, v in dup_target.items())}.",
        f"- Removed them, keeping the first occurrence: {n_before:,} → **{len(df):,} rows** "
        f"({n_dups / n_before:.3%} of the data).",
        "- Rationale: the dataset has no client ID, so identical rows across all attributes "
        "(including call duration and daily economic indicators) are almost certainly repeated "
        "records. Keeping them would put copies in both train and test splits and slightly "
        "inflate evaluation scores.", "",
    ]

    # ------------------------------------------------------------------ 3. 'unknown' values
    section("3. 'unknown' VALUES IN CATEGORICAL COLUMNS")
    overall_rate = (df["y"] == "yes").mean()
    rows = []
    for col in CATEGORICAL_COLUMNS:
        is_unknown = df[col] == "unknown"
        n = int(is_unknown.sum())
        if n == 0:
            continue
        rows.append({
            "column": col,
            "unknown_count": n,
            "unknown_pct": f"{n / len(df):.2%}",
            "subscription_rate_if_unknown": f"{(df.loc[is_unknown, 'y'] == 'yes').mean():.1%}",
            "subscription_rate_if_known": f"{(df.loc[~is_unknown, 'y'] == 'yes').mean():.1%}",
        })
    unknown_df = pd.DataFrame(rows).set_index("column")
    print(unknown_df.to_string())
    print(f"(overall subscription rate: {overall_rate:.1%})")

    any_unknown = (df[CATEGORICAL_COLUMNS] == "unknown").any(axis=1)
    housing_loan_same = ((df["housing"] == "unknown") == (df["loan"] == "unknown")).all()
    default_counts = df["default"].value_counts().to_dict()
    print(f"Rows with at least one 'unknown': {int(any_unknown.sum()):,} ({any_unknown.mean():.1%})")
    print(f"housing/loan 'unknown' on exactly the same rows: {housing_loan_same}")
    print(f"default value counts: {default_counts}")
    # Decision: keep 'unknown' as its own category (no deletion, no imputation).

    md += [
        "## 3. `unknown` values in categorical columns", "",
        md_table(unknown_df), "",
        f"Overall subscription rate after de-duplication: {overall_rate:.1%}.", "",
        f"- **{int(any_unknown.sum()):,} rows ({any_unknown.mean():.1%})** contain at least one "
        "`unknown`, so deleting them would discard a large share of the data.",
        f"- `housing` and `loan` are `unknown` on {'exactly the same' if housing_loan_same else 'different'} "
        "rows, suggesting a single missing source record rather than independent gaps.",
        f"- `default` is effectively binary *known-no* vs *unknown*: "
        f"{', '.join(f'{k}={v:,}' for k, v in default_counts.items())}. Subscription is far lower when "
        "`default` is unknown, so the missingness itself is informative.",
        "- **Treatment:** `unknown` is kept as a separate category level (no deletion, no imputation). "
        "The missingness is not random (it correlates with the target), imputing the mode would "
        "erase that signal, and one-hot or native categorical encoding handles an extra level "
        "naturally. The documentation explicitly lists this as a valid option.",
        "- Note for modelling: `default = yes` has only "
        f"{default_counts.get('yes', 0)} rows; it carries almost no information on its own.", "",
    ]

    # ------------------------------------------------------------------ 4. Numeric checks
    section("4. NUMERIC FEATURES: OUTLIERS & SPECIAL VALUES")
    outlier_rows = []
    for col in OUTLIER_CHECK_COLUMNS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        outlier_rows.append({
            "feature": col, "min": df[col].min(), "median": df[col].median(),
            "max": df[col].max(), "IQR_upper_fence": round(high, 1),
            "n_beyond_fences": n_out, "pct": f"{n_out / len(df):.2%}",
        })
    outlier_df = pd.DataFrame(outlier_rows).set_index("feature")
    print(outlier_df.to_string())

    # pdays sentinel
    pdays_999 = df["pdays"] == PDAYS_SENTINEL
    n_999 = int(pdays_999.sum())
    real_pdays = df.loc[~pdays_999, "pdays"]
    n_999_but_previous = int((pdays_999 & (df["previous"] > 0)).sum())
    poutcome_by_pdays = pd.crosstab(df["poutcome"], pd.Series(np.where(pdays_999, "pdays=999", "pdays<999"),
                                                           name="pdays", index=df.index))
    print(f"\npdays == {PDAYS_SENTINEL}: {n_999:,} rows ({n_999 / len(df):.1%})")
    print(f"pdays excluding sentinel: min={real_pdays.min()}, median={real_pdays.median()}, "
          f"max={real_pdays.max()}")
    print(f"pdays == 999 but previous > 0: {n_999_but_previous:,}")
    print(poutcome_by_pdays.to_string())

    # duration special values
    dur0 = df["duration"] == 0
    dur0_target = df.loc[dur0, "y"].value_counts().to_dict()
    print(f"\nduration == 0: {int(dur0.sum())} rows, target {dur0_target}")

    # plausibility of ranges
    age_ok = df["age"].between(17, 100).all()
    campaign_ok = (df["campaign"] >= 1).all()
    print(f"age within [17, 100]: {age_ok}; campaign >= 1: {campaign_ok}")

    # Treatment: explicit indicator; keep raw pdays for transparency. Use `previous` too, since
    # pdays is 999 for some clients who were contacted before (see inconsistency below).
    df["previously_contacted"] = ((~pdays_999) | (df["previous"] > 0)).astype(int)
    n_prev = int(df["previously_contacted"].sum())
    print(f"previously_contacted = 1: {n_prev:,} rows")

    md += [
        "## 4. Numeric features: outliers and special values", "",
        "IQR rule (values beyond Q1 − 1.5·IQR or Q3 + 1.5·IQR):", "",
        md_table(outlier_df), "",
        f"- **age** ({df['age'].min()}–{df['age'].max()}): high values are retirees, all plausible. Kept.",
        f"- **duration** (0–{df['duration'].max():,} s, max ≈ {df['duration'].max() / 60:.0f} min): "
        "long-tailed but plausible call lengths. Kept. "
        f"{int(dur0.sum())} calls have `duration = 0`, all with y = "
        f"{', '.join(f'{k} ({v})' for k, v in dur0_target.items())}, as the documentation predicts.",
        f"- **campaign** (1–{df['campaign'].max()}): heavy right tail (repeat calls within one campaign). "
        "Plausible, kept.",
        "- **previous**: the IQR is 0 because most clients were never contacted before, so the IQR "
        "rule flags every non-zero value. Not true outliers. Kept.",
        "- No values were removed or capped. The extremes are real behaviour, not errors, and the "
        "AutoML candidates (gradient-boosted trees, random forests) are insensitive to monotone "
        "scale. If linear models are used, consider a log transform of `duration`/`campaign` at "
        "training time.", "",
        f"### `pdays` sentinel value ({PDAYS_SENTINEL})", "",
        f"- `pdays = {PDAYS_SENTINEL}` means *not previously contacted*: **{n_999:,} rows "
        f"({n_999 / len(df):.1%})**. Genuine values range {real_pdays.min()}–{real_pdays.max()} days "
        f"(median {real_pdays.median():.0f}), so treating 999 as a number would distort any "
        "distance- or scale-based model.",
        f"- Inconsistency found: **{n_999_but_previous:,} rows** have `pdays = {PDAYS_SENTINEL}` yet "
        "`previous > 0` and `poutcome = failure`, i.e. the client *was* contacted before but the "
        "day count was not recorded. So `pdays = 999` does not reliably mean *never contacted*.",
        f"- **Treatment:** added a binary indicator `previously_contacted` = 1 if `pdays != {PDAYS_SENTINEL}` "
        f"**or** `previous > 0` (**{n_prev:,} rows**, {n_prev / len(df):.1%}). The raw `pdays` column is "
        "kept unchanged so the sentinel can be handled per model (trees split on it cleanly; for "
        "linear models, use the indicator and neutralise 999).", "",
        md_table(poutcome_by_pdays), "",
        f"- Range checks: age within [17, 100]: {check(age_ok)}; campaign ≥ 1: {check(campaign_ok)}.", "",
    ]

    # ------------------------------------------------------------------ 5. Target
    section("5. TARGET ENCODING")
    df["y"] = df["y"].map({"no": 0, "yes": 1}).astype(int)
    counts = df["y"].value_counts().sort_index()
    print(f"y mapped: no -> 0, yes -> 1\n{counts.to_string()}")
    print(f"Positive rate: {df['y'].mean():.2%}; imbalance ratio {counts[0] / counts[1]:.1f}:1")

    md += [
        "## 5. Target encoding", "",
        "`y` mapped to integers: `no` → 0, `yes` → 1.", "",
        "| y | count | share |", "|---|---|---|",
        f"| 0 (no) | {counts[0]:,} | {counts[0] / len(df):.2%} |",
        f"| 1 (yes) | {counts[1]:,} | {counts[1] / len(df):.2%} |", "",
        f"The classes are imbalanced (≈{counts[0] / counts[1]:.1f} : 1). Use stratified splits and "
        "imbalance-aware metrics (ROC-AUC, PR-AUC, F1) rather than accuracy.", "",
    ]

    # ------------------------------------------------------------------ 6. Save
    section("6. SAVE")
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved {df.shape[0]:,} rows × {df.shape[1]} cols -> {CLEAN_PATH.relative_to(PROJECT_ROOT)}")

    md += [
        "## 6. Output", "",
        f"- File: `data/cleaned_bank_data.csv` (comma-delimited), "
        f"**{df.shape[0]:,} rows × {df.shape[1]} columns**.",
        "- Columns: the 20 original features, `y` (0/1), and the new indicator `previously_contacted`.",
        "- `duration` is **retained for EDA only**. It is recorded after the call and leaks the "
        "outcome, so it must be excluded for a realistic model. "
        "`src.data.load_clean()` drops it by default (`drop_leaky=False` keeps it for benchmarking).", "",
        "## Cleaning steps at a glance", "",
        "| Step | Action | Rows affected |", "|---|---|---|",
        f"| Duplicates | Removed exact duplicate rows | −{n_dups} |",
        f"| `unknown` values | Kept as an explicit category in 6 columns | {int(any_unknown.sum()):,} rows contain ≥1 |",
        f"| `pdays = {PDAYS_SENTINEL}` | Added `previously_contacted` indicator (pdays or previous); raw value kept | {n_999:,} |",
        "| Outliers | Reviewed with IQR rule; none removed (all plausible) | 0 |",
        "| Target | `y` mapped no/yes → 0/1 | all |",
        "| `duration` | Retained, flagged as leaky (drop at training time) | n/a |",
        "",
    ]

    summary = "\n".join(md)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")

    section("MARKDOWN SUMMARY")
    print(summary)
    print(f"(also written to {SUMMARY_PATH.relative_to(PROJECT_ROOT)})")


if __name__ == "__main__":
    main()
