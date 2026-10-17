# Data Preprocessing Summary

Source: `data/raw/bank-additional-full.csv` (semicolon-delimited)

## 1. Verification checks

| Check | Expected | Observed | Result |
|---|---|---|---|
| Dimensions | 41,188 rows × 21 cols | 41,188 rows × 21 cols | Pass |
| Column names & order | 21 documented attributes | 21 columns, matching | Pass |
| Numeric dtypes | 10 numeric features | 10 numeric (5 int64, 5 float64) | Pass |
| Categorical dtypes | 10 categorical features + target | all string | Pass |
| Null values (NaN) | none documented | 0 | Pass |
| Target values | `no`, `yes` | `no`, `yes` | Pass |

Missing values are not stored as NaN: the documentation states they are coded as the string `unknown` in categorical columns (see section 3).

### Numeric summary statistics (raw)

| feature | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| age | 41188.0 | 40.02 | 10.42 | 17.0 | 32.0 | 38.0 | 47.0 | 98.0 |
| duration | 41188.0 | 258.29 | 259.28 | 0.0 | 102.0 | 180.0 | 319.0 | 4918.0 |
| campaign | 41188.0 | 2.57 | 2.77 | 1.0 | 1.0 | 2.0 | 3.0 | 56.0 |
| pdays | 41188.0 | 962.48 | 186.91 | 0.0 | 999.0 | 999.0 | 999.0 | 999.0 |
| previous | 41188.0 | 0.17 | 0.49 | 0.0 | 0.0 | 0.0 | 0.0 | 7.0 |
| emp.var.rate | 41188.0 | 0.08 | 1.57 | -3.4 | -1.8 | 1.1 | 1.4 | 1.4 |
| cons.price.idx | 41188.0 | 93.58 | 0.58 | 92.2 | 93.08 | 93.75 | 93.99 | 94.77 |
| cons.conf.idx | 41188.0 | -40.5 | 4.63 | -50.8 | -42.7 | -41.8 | -36.4 | -26.9 |
| euribor3m | 41188.0 | 3.62 | 1.73 | 0.63 | 1.34 | 4.86 | 4.96 | 5.04 |
| nr.employed | 41188.0 | 5167.04 | 72.25 | 4963.6 | 5099.1 | 5191.0 | 5228.1 | 5228.1 |

## 2. Duplicate rows

- Found **12 exact duplicate rows** (all 21 columns identical); target among the duplicates: no=11, yes=1.
- Removed them, keeping the first occurrence: 41,188 → **41,176 rows** (0.029% of the data).
- Rationale: the dataset has no client ID, so identical rows across all attributes (including call duration and daily economic indicators) are almost certainly repeated records. Keeping them would put copies in both train and test splits and slightly inflate evaluation scores.

## 3. `unknown` values in categorical columns

| column | unknown_count | unknown_pct | subscription_rate_if_unknown | subscription_rate_if_known |
|---|---|---|---|---|
| job | 330 | 0.80% | 11.2% | 11.3% |
| marital | 80 | 0.19% | 15.0% | 11.3% |
| education | 1730 | 4.20% | 14.5% | 11.1% |
| default | 8596 | 20.88% | 5.2% | 12.9% |
| housing | 990 | 2.40% | 10.8% | 11.3% |
| loan | 990 | 2.40% | 10.8% | 11.3% |

Overall subscription rate after de-duplication: 11.3%.

- **10,698 rows (26.0%)** contain at least one `unknown`, so deleting them would discard a large share of the data.
- `housing` and `loan` are `unknown` on exactly the same rows, suggesting a single missing source record rather than independent gaps.
- `default` is effectively binary *known-no* vs *unknown*: no=32,577, unknown=8,596, yes=3. Subscription is far lower when `default` is unknown, so the missingness itself is informative.
- **Treatment:** `unknown` is kept as a separate category level (no deletion, no imputation). The missingness is not random (it correlates with the target), imputing the mode would erase that signal, and one-hot or native categorical encoding handles an extra level naturally. The documentation explicitly lists this as a valid option.
- Note for modelling: `default = yes` has only 3 rows; it carries almost no information on its own.

## 4. Numeric features: outliers and special values

IQR rule (values beyond Q1 − 1.5·IQR or Q3 + 1.5·IQR):

| feature | min | median | max | IQR_upper_fence | n_beyond_fences | pct |
|---|---|---|---|---|---|---|
| age | 17 | 38.0 | 98 | 69.5 | 468 | 1.14% |
| duration | 0 | 180.0 | 4918 | 644.5 | 2963 | 7.20% |
| campaign | 1 | 2.0 | 56 | 6.0 | 2406 | 5.84% |
| previous | 0 | 0.0 | 7 | 0.0 | 5625 | 13.66% |

- **age** (17–98): high values are retirees, all plausible. Kept.
- **duration** (0–4,918 s, max ≈ 82 min): long-tailed but plausible call lengths. Kept. 4 calls have `duration = 0`, all with y = no (4), as the documentation predicts.
- **campaign** (1–56): heavy right tail (repeat calls within one campaign). Plausible, kept.
- **previous**: the IQR is 0 because most clients were never contacted before, so the IQR rule flags every non-zero value. Not true outliers. Kept.
- No values were removed or capped. The extremes are real behaviour, not errors, and the AutoML candidates (gradient-boosted trees, random forests) are insensitive to monotone scale. If linear models are used, consider a log transform of `duration`/`campaign` at training time.

### `pdays` sentinel value (999)

- `pdays = 999` means *not previously contacted*: **39,661 rows (96.3%)**. Genuine values range 0–27 days (median 6), so treating 999 as a number would distort any distance- or scale-based model.
- Inconsistency found: **4,110 rows** have `pdays = 999` yet `previous > 0` and `poutcome = failure`, i.e. the client *was* contacted before but the day count was not recorded. So `pdays = 999` does not reliably mean *never contacted*.
- **Treatment:** added a binary indicator `previously_contacted` = 1 if `pdays != 999` **or** `previous > 0` (**5,625 rows**, 13.7%). The raw `pdays` column is kept unchanged so the sentinel can be handled per model (trees split on it cleanly; for linear models, use the indicator and neutralise 999).

| poutcome | pdays<999 | pdays=999 |
|---|---|---|
| failure | 142 | 4110 |
| nonexistent | 0 | 35551 |
| success | 1373 | 0 |

- Range checks: age within [17, 100]: Pass; campaign ≥ 1: Pass.

## 5. Target encoding

`y` mapped to integers: `no` → 0, `yes` → 1.

| y | count | share |
|---|---|---|
| 0 (no) | 36,537 | 88.73% |
| 1 (yes) | 4,639 | 11.27% |

The classes are imbalanced (≈7.9 : 1). Use stratified splits and imbalance-aware metrics (ROC-AUC, PR-AUC, F1) rather than accuracy.

## 6. Output

- File: `data/cleaned_bank_data.csv` (comma-delimited), **41,176 rows × 22 columns**.
- Columns: the 20 original features, `y` (0/1), and the new indicator `previously_contacted`.
- `duration` is **retained for EDA only**. It is recorded after the call and leaks the outcome, so it must be excluded for a realistic model. `src.data.load_clean()` drops it by default (`drop_leaky=False` keeps it for benchmarking).

## Cleaning steps at a glance

| Step | Action | Rows affected |
|---|---|---|
| Duplicates | Removed exact duplicate rows | −12 |
| `unknown` values | Kept as an explicit category in 6 columns | 10,698 rows contain ≥1 |
| `pdays = 999` | Added `previously_contacted` indicator (pdays or previous); raw value kept | 39,661 |
| Outliers | Reviewed with IQR rule; none removed (all plausible) | 0 |
| Target | `y` mapped no/yes → 0/1 | all |
| `duration` | Retained, flagged as leaky (drop at training time) | n/a |
