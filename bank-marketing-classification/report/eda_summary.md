# Exploratory Data Analysis

Dataset: `data/cleaned_bank_data.csv` (41,176 clients × 22 columns), the output of
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

| feature | mean | std | min | median | max | median_no | median_yes |
|---|---|---|---|---|---|---|---|
| age | 40.02 | 10.42 | 17 | 38 | 98 | 38 | 37 |
| duration | 258.32 | 259.31 | 0 | 180 | 4,918 | 164 | 449 |
| campaign | 2.57 | 2.77 | 1 | 2 | 56 | 2 | 2 |
| pdays | 962.46 | 186.94 | 0 | 999 | 999 | 999 | 999 |
| previous | 0.17 | 0.49 | 0 | 0 | 7 | 0 | 0 |
| previously_contacted | 0.14 | 0.34 | 0 | 0 | 1 | 0 | 0 |
| emp.var.rate | 0.08 | 1.57 | -3.4 | 1.1 | 1.4 | 1.1 | -1.8 |
| cons.price.idx | 93.58 | 0.58 | 92.2 | 93.75 | 94.77 | 93.92 | 93.2 |
| cons.conf.idx | -40.5 | 4.63 | -50.8 | -41.8 | -26.9 | -41.8 | -40.4 |
| euribor3m | 3.62 | 1.73 | 0.63 | 4.86 | 5.04 | 4.86 | 1.27 |
| nr.employed | 5,167.03 | 72.25 | 4,963.6 | 5,191 | 5,228.1 | 5,195.8 | 5,099.1 |

Observations:

- **Clients:** median age 38 (range 17–98).
  Subscribers and non-subscribers have almost the same median age; the age effect is
  non-linear (see section 4).
- **Campaign effort:** median 2 contacts per client this campaign, with a
  long tail up to 56. Only 13.7% of clients had any
  prior contact (`previously_contacted`), so `pdays` and `previous` are mostly constant.
- **Duration:** median call 7.5 min for subscribers vs 2.7 min for
  non-subscribers (2.7× longer).
- **Macro context:** `euribor3m` is bimodal (clusters near 1% and near 4.9%), and its
  median is 1.27% for subscribers vs 4.86% for non-subscribers. These indicators
  vary by contact date, not by client, so they record *when* a client was called.

## 2. Key correlations (Figure 2)

**With the target.** The strongest linear associations with `y` are
`duration` (r = +0.41), `nr.employed` (r = -0.35), `pdays` (r = -0.32), `euribor3m` (r = -0.31), `emp.var.rate` (r = -0.30).
Every macro indicator except consumer confidence is negatively correlated with subscription:
clients said yes more often when rates, employment and prices were low.

**Among features (multicollinearity).** The macro indicators are close to redundant:
`emp.var.rate` ↔ `euribor3m` r = +0.97; `euribor3m` ↔ `nr.employed` r = +0.95; `emp.var.rate` ↔ `nr.employed` r = +0.91; `emp.var.rate` ↔ `cons.price.idx` r = +0.78.
`previous` and `previously_contacted` are also strongly linked (r = +0.88)
by construction, as are `pdays` and `previous` (r = -0.59). This matters for linear or logistic models (unstable coefficients; consider
keeping one or two indicators, or regularising) and for interpreting feature importances
(importance is split across correlated columns). Tree ensembles, as used by FLAML, are robust to it.

## 3. Class imbalance (Figure 1)

Only **4,639 of 41,176 clients (11.3%) subscribed**, a ratio of about **7.9 : 1**.

**Statistical impact**

- **Accuracy is misleading:** predicting "no" for everyone scores **88.7% accuracy** while finding
  zero subscribers. Use ROC-AUC, **PR-AUC** (its no-skill baseline equals the positive rate, 0.113),
  F1, recall and precision for the positive class.
- **Learning bias:** unweighted models minimise overall error by under-predicting the minority class,
  and default probability outputs centre near the base rate. Remedies: class weights
  (`class_weight="balanced"` / `scale_pos_weight ≈ 7.9`), tuning the decision threshold on
  validation data instead of using 0.5, and checking probability calibration.
- **Evaluation stability:** a test set of 20% holds only about 927 positives, so
  metric estimates for the positive class are noisy. Use **stratified** train/test splits and
  stratified k-fold cross-validation. Resampling (SMOTE, undersampling), if used, must be applied only
  inside the training folds.

**Operational impact**

- Calling at random converts 11.3% of contacts: about **8.9 calls per subscription**.
  A model is valuable if it ranks clients so the top-scored segment converts well above that
  base rate (measure with lift / cumulative gains, precision@k).
- The costs of the two errors are unequal: a false positive costs one unnecessary call, while a false
  negative loses a deposit. The decision threshold should come from that cost trade-off and from
  call-centre capacity, not from the default 0.5.

## 4. What predicts subscription

### Univariate strength: numeric features

ROC-AUC of each feature used alone as a score (0.5 = no signal, symmetric in direction).

| Feature | Pearson r with y | Univariate ROC-AUC | Direction |
|---|---|---|---|
| duration | +0.405 | 0.818 | higher → more yes |
| nr.employed | -0.355 | 0.749 | higher → less yes |
| euribor3m | -0.308 | 0.743 | higher → less yes |
| emp.var.rate | -0.298 | 0.717 | higher → less yes |
| cons.price.idx | -0.136 | 0.611 | higher → less yes |
| previous | +0.230 | 0.609 | higher → more yes |
| previously_contacted | +0.194 | 0.605 | higher → more yes |
| pdays | -0.325 | 0.597 | higher → less yes |
| campaign | -0.066 | 0.555 | higher → less yes |
| cons.conf.idx | +0.055 | 0.537 | higher → more yes |
| age | +0.030 | 0.511 | higher → less yes |

### Univariate strength: categorical features

Cramér's V (0 = independent of `y`, 1 = perfectly associated). Extremes ignore groups with n < 100.

| Feature | Cramér's V | Lowest rate (n ≥ 100) | Highest rate (n ≥ 100) |
|---|---|---|---|
| poutcome | 0.321 | nonexistent (8.8%) | success (65.1%) |
| month | 0.275 | may (6.4%) | mar (50.5%) |
| job | 0.153 | blue-collar (6.9%) | student (31.4%) |
| contact | 0.145 | telephone (5.2%) | cellular (14.7%) |
| default | 0.099 | unknown (5.2%) | no (12.9%) |
| education | 0.068 | basic.9y (7.8%) | unknown (14.5%) |
| marital | 0.055 | married (10.2%) | single (14.0%) |
| day_of_week | 0.025 | mon (10.0%) | thu (12.1%) |
| housing | 0.012 | unknown (10.8%) | yes (11.6%) |
| loan | 0.005 | unknown (10.8%) | no (11.3%) |

### Campaign attributes (most predictive group)

- **`duration` is the single strongest feature** (AUC 0.82):
  subscription goes from 0.0% for calls of ≤1 min to 59.2% for calls
  over 17 min (Figure 4b). Because it is known only after the call, it is a *consequence* of the
  outcome, not a predictor for targeting, and is **excluded from the realistic model**. It can be
  used as a benchmark upper bound.
- **Previous campaign outcome** (`poutcome`, V = 0.32) is the
  strongest pre-call signal: success 65.1%, failure 14.2%, nonexistent 8.8% (Figure 3).
  Any prior contact (`previously_contacted`) triples the rate: 26.6% vs
  8.8%. Only 5,625 clients fall in this group, so it is a high-value but small segment.
- **Contact month** (V = 0.27): mar 50.5%, dec 48.9%, sep 44.9%, oct 43.9%
  versus 6.4% in May, which accounts for 33.4% of all calls. The
  high-rate months are low-volume, so this may reflect campaign scheduling and targeting rather than
  true seasonality.
- **Contact channel:** cellular 14.7% vs telephone 5.2%.
- **Contact frequency:** the rate falls as contacts within this campaign rise:
  1 call 13.0%, 2 calls 11.5%, 3 calls 10.7%, 4–5 calls 8.7%, 6–10 calls 6.3%, >10 calls 3.1%. Repeated
  calls have diminishing returns.

### Macroeconomic factors

- `nr.employed`, `euribor3m` and `emp.var.rate` have univariate AUCs of
  0.75, 0.74 and
  0.72, making them the strongest legitimate numeric predictors.
- In the lowest Euribor quintile (0.63–1.30%) **29.7%** subscribed, versus
  3.6%–5.5% in the top three quintiles (Figure 4c).
- Quarters with negative employment variation (e.g. -3.4: 42.4%) convert at
  several times the rate of growth quarters (+1.1: 3.1%, +1.4: 5.3%) (Figure 4d).
- Interpretation: these indicators change with the contact date, not the client, so they act as
  time-period effects. A plausible explanation (not testable from this data alone) is that the
  low-rate, recessionary period coincided with smaller, more targeted campaigns and with safe deposits
  becoming more attractive to clients. Either way, they predict well within
  this 2008–2010 window but will **not transfer automatically to different economic regimes**, so the
  model should be monitored and retrained when conditions change. Because the data is ordered by date,
  a time-based validation split would test this directly.

### Client demographics

- **Job** (V = 0.15): student 31.4%, retired 25.3% are the most
  responsive groups; blue-collar 6.9%, services 8.1% are the least (Figure 3).
- **Age** is U-shaped, which is why its linear correlation is near zero (r = +0.03):
  ≤24: 24.0%, 25–34: 12.2%, 35–44: 8.7%, 45–54: 8.6%, 55–64: 13.6%, 65+: 47.3%. Tree models or age bands capture this.
- **Credit default status:** 12.9% when `default = no` vs 5.2% when
  `unknown`. The missingness itself is informative, which supports keeping `unknown` as a category.
- **Education** has a modest effect (V = 0.07): university degree
  13.7% vs basic.9y 7.8%.
- **Weak or no signal:** `housing` (V = 0.012), `loan`
  (V = 0.005), `day_of_week` (V = 0.025)
  and `marital` (V = 0.055). These are candidates for removal
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
