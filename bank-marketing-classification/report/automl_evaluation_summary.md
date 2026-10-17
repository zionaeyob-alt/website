# AutoML Training and Evaluation

Script: `src/train_and_evaluate.py` · Framework: FLAML 2.7.0 ·
Data: `data/cleaned_bank_data.csv` (41,176 rows)

## 1. Setup

| Item | Setting |
|---|---|
| Split | Stratified 80/20, `random_state=42`: 32,940 train (11.27% positive), 8,236 test (11.27% positive) |
| Primary model | 20 features, `duration` **dropped** (realistic pre-call targeting) |
| Benchmark model | 21 features, `duration` **kept** (leakage demonstration) |
| Task / metric | Binary classification, optimise ROC-AUC |
| Validation | 5-fold stratified cross-validation inside the training set |
| Search space | LightGBM, XGBoost, Random Forest, Logistic Regression (one-hot + scaled, custom learner) |
| Budget | 180 s per model (identical for both, so the comparison is fair) |
| Threshold tuning | F1-maximising threshold chosen on out-of-fold predictions of the best config (5 stratified folds of the training set). The test set is used only for final evaluation |
| Saved artifacts | `models/automl_primary.pkl`, `models/automl_benchmark.pkl`, `models/model_metadata.json` |

The categorical features are passed to FLAML as pandas `category` columns: LightGBM and XGBoost use
native categorical splits, and the logistic regression one-hot encodes them. FLAML's search is
time-based, so a re-run can pick slightly different hyperparameters; the split and seeds are fixed.

## 2. Primary model: search results

**Chosen model family: LightGBM**, best 5-fold CV ROC-AUC
**0.8018**, found after 72 s
(trial 42 of 71).

Best hyperparameters: `n_estimators` = 1240, `num_leaves` = 4, `min_child_samples` = 82, `learning_rate` = 0.09806, `log_max_bin` = 6, `colsample_bytree` = 0.8622, `reg_alpha` = 0.007111, `reg_lambda` = 38.4.

### Search trajectory across algorithms

| Learner | Trials | Best 5-fold CV ROC-AUC | Best config |
|---|---|---|---|
| LightGBM | 41 | 0.8018 | `n_estimators` = 1240, `num_leaves` = 4, `min_child_samples` = 82, `learning_rate` = 0.09806, `log_max_bin` = 6, `colsample_bytree` = 0.8622, `reg_alpha` = 0.007111, `reg_lambda` = 38.4 |
| Random Forest | 26 | 0.7918 | `n_estimators` = 7, `max_features` = 0.2267, `max_leaves` = 24, `criterion` = entropy |
| Logistic Regression | 2 | 0.7898 | `C` = 3.981 |
| XGBoost | 2 | 0.7441 | `n_estimators` = 5, `max_leaves` = 4, `min_child_weight` = 0.6257, `learning_rate` = 0.06767, `subsample` = 1, `colsample_bylevel` = 0.9569, `colsample_bytree` = 1, `reg_alpha` = 0.01839, `reg_lambda` = 6.322 |

FLAML's cost-aware search starts every learner from a cheap configuration and gives more trials to
learners that improve fastest per second of compute, so trial counts differ by learner
(71 trials in total) and also between runs, because the budget is wall-clock time.
Among learners searched at least 5 times, LightGBM led Random Forest by 0.0099 CV ROC-AUC. These gaps are small: the main signal (macro indicators, month, contact type, contact history) is captured by every well-searched learner. **Under-explored:** Logistic Regression (2 trials, best 0.7898), XGBoost (2 trials, best 0.7441). Their scores are lower bounds from a few configurations, not a fair verdict on the algorithm; a longer budget would be needed to compare them properly.

Each time the overall best improved:

| Trial | Time (s) | Learner | CV ROC-AUC |
|---|---|---|---|
| 0 | 0.3 | LightGBM | 0.7488 |
| 1 | 0.7 | LightGBM | 0.7676 |
| 2 | 0.9 | LightGBM | 0.7866 |
| 3 | 1.2 | LightGBM | 0.7918 |
| 4 | 1.8 | LightGBM | 0.7950 |
| 7 | 3.1 | LightGBM | 0.8001 |
| 29 | 33.7 | LightGBM | 0.8006 |
| 31 | 36.2 | LightGBM | 0.8008 |
| 40 | 63.4 | LightGBM | 0.8011 |
| 42 | 72.3 | LightGBM | 0.8018 |

## 3. Test-set evaluation (20% holdout, 8,236 clients, 928 subscribers)

| Model | Threshold | Accuracy | Balanced Acc. | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| Primary (no duration) | 0.5 | 0.899 | 0.614 | 0.636 | 0.247 | 0.356 | 0.810 | 0.479 |
| Primary (no duration) | 0.202 (tuned) | 0.880 | 0.758 | 0.474 | 0.601 | 0.530 | 0.810 | 0.479 |
| Benchmark (with duration) | 0.5 | 0.919 | 0.765 | 0.665 | 0.566 | 0.612 | 0.953 | 0.675 |
| Benchmark (with duration) | 0.326 (tuned) | 0.911 | 0.854 | 0.577 | 0.780 | 0.664 | 0.953 | 0.675 |

- **CV vs test:** the primary model's test ROC-AUC (0.8098) is close to its CV
  estimate (0.8018), so the search did not overfit the validation folds.
- **ROC-AUC 0.810:** a randomly chosen subscriber is ranked above a randomly chosen
  non-subscriber 81% of the time.
- **PR-AUC 0.479** against a no-skill baseline of 0.113, i.e.
  4.3× better than random targeting across all recall levels.
- **Accuracy is not informative here:** the default threshold scores 89.9%
  accuracy, barely above the 88.7% obtained by predicting "no" for everyone.

### Default (0.5) vs tuned threshold (0.202)

With an 11% positive rate, predicted probabilities rarely exceed 0.5, so the default threshold
calls very few clients:

| Primary model | Clients flagged | Subscribers found (TP) | Missed subscribers (FN) | Wasted calls (FP) | Precision | Recall |
|---|---|---|---|---|---|---|
| Threshold 0.5 | 360 | 229 | 699 | 131 | 63.6% | 24.7% |
| Threshold 0.202 | 1,176 | 558 | 370 | 618 | 47.4% | 60.1% |

Lowering the threshold to 0.202 finds **+329 more subscribers** at the cost of
**+487 extra calls** to non-subscribers, raising F1 from 0.356 to
0.530 and balanced accuracy from 0.614 to
0.758.

## 4. Interpretation in a banking context

**The two errors have very different costs.**

- A **false positive** is a call to a client who will not subscribe. Its cost is agent time and
  telephony (a few minutes per call), plus a small risk of annoying the client.
- A **false negative** is a client who would have subscribed but is never called. The cost is the
  lost margin on a term deposit over its whole life, typically far larger than the cost of one call.

So when call capacity allows, the bank should **favour recall over precision**, i.e. a threshold
below the F1-optimal one. The F1 threshold is a neutral starting point; the operating threshold
should be set from the actual cost per call, the expected margin per deposit, and the number of
calls the team can make. For example, if a deposit is worth 10× the cost of a call, calling anyone
with a predicted probability above about 1/11 ≈ 0.09 has positive expected value.

**Ranking view (how campaigns are actually run).** Sorting test clients by predicted probability:

| Call the top… | Conversion rate of those calls | Share of all subscribers reached | Lift vs random |
|---|---|---|---|
| 10% | 51.3% | 45.6% | 4.6× |
| 20% | 37.8% | 67.0% | 3.4× |
| 30% | 27.7% | 73.8% | 2.5× |

Random calling converts 11.3%. Calling only the top-scored 20% of clients reaches
67% of all subscribers with 3.4× the conversion rate, i.e. roughly 80% fewer
calls for 67% of the business.

**What drives the primary model** (permutation importance, Figure 7):
`emp.var.rate` (0.038), `nr.employed` (0.029), `euribor3m` (0.016), `contact` (0.016), `month` (0.014), `day_of_week` (0.004), `pdays` (0.003), `cons.price.idx` (0.003).
This agrees with the EDA: macro conditions, contact history, timing and channel matter most;
personal demographics matter less.

## 5. Impact of `duration` (data leakage)

| Metric (tuned threshold) | Primary (no duration) | Benchmark (with duration) | Difference |
|---|---|---|---|
| ROC-AUC | 0.810 | 0.953 | +0.143 |
| PR-AUC | 0.479 | 0.675 | +0.196 |
| F1 | 0.530 | 0.664 | +0.133 |
| Precision | 0.474 | 0.577 | +0.103 |
| Recall | 0.601 | 0.780 | +0.179 |

- The benchmark (XGBoost, CV ROC-AUC 0.9492)
  looks far better: PR-AUC rises by +0.196 (1.4×) and F1 by +0.133.
- `duration` is the benchmark's **#1 most important feature**: shuffling it alone lowers test
  ROC-AUC by 0.226.
- This gain is not usable. Call duration is only known once the call has ended, when the outcome is
  already known, and a long call is partly a *result* of the client being interested. A model that
  depends on it cannot rank clients before calling them.
- The benchmark therefore shows how much leakage would **overstate** performance. Reporting it as the
  model's accuracy would mislead stakeholders. The primary model's results are the honest estimate.

## 6. Limitations

- Random 80/20 split: the data is ordered in time, and macro indicators act as time-period proxies,
  so a time-based split would give a more conservative estimate for future campaigns.
- Hyperparameters were tuned under a 180-second budget; a longer search may add a little.
- Probabilities were not calibrated. If expected-value thresholds are used operationally, check
  calibration first (e.g. with a reliability curve).
