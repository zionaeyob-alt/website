# Bank Marketing – Term Deposit Classification

Predict whether a client subscribes to a term deposit (`y`) from the
[UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
(Moro et al., 2014).

## Layout

```
bank-marketing-classification/
├── data/
│   ├── raw/            # original UCI files (committed, read-only)
│   ├── cleaned_bank_data.csv  # output of src/data_preprocessing.py
│   └── processed/      # cleaned / feature-engineered data (git-ignored)
├── notebooks/          # exploratory Jupyter notebooks
├── scripts/            # runnable scripts (verify_env.py, fetch_data.py)
├── src/                # reusable code (src/data.py: paths + loaders)
├── figures/            # high-resolution plots (save at dpi=300)
├── models/             # saved model artifacts (git-ignored)
├── report/             # report drafts
├── requirements.txt        # direct dependencies (minimum versions)
└── requirements-lock.txt   # exact versions verified working (Python 3.13)
```

## Data (`data/raw/`)

| File | Rows | Features | Notes |
|---|---|---|---|
| `bank-additional-full.csv` | 41,188 | 20 | **Primary dataset**: adds 5 macro-economic indicators; May 2008 – Nov 2010 |
| `bank-additional.csv` | 4,119 | 20 | 10% random sample of the above, for fast iteration |
| `bank-full.csv` | 45,211 | 16 | Older version (has `balance`, `day`; no macro indicators) |
| `bank.csv` | 4,521 | 16 | 10% random sample of `bank-full.csv` |

Feature descriptions are in `bank-additional-names.txt` and `bank-names.txt`.
All CSVs use `;` as the separator. About 11% of rows are positive, so the classes are imbalanced: use
ROC-AUC, PR-AUC or F1 rather than accuracy.

> **Leakage:** `duration` is only known after the call ends. `src.data.load_raw()`
> drops it by default; pass `drop_leaky=False` only for benchmark comparisons.

To re-download the raw files: `python scripts/fetch_data.py`.

## Setup

Requires Python 3.10+ (verified on 3.13).

```bash
cd bank-marketing-classification
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt      # or requirements-lock.txt for exact versions

# optional: register the venv as a Jupyter kernel
python -m ipykernel install --user --name bank-marketing --display-name "Bank Marketing (.venv)"
```

macOS only: LightGBM needs OpenMP: `brew install libomp`.

## Verify

```bash
python scripts/verify_env.py          # imports, data row counts, 10s FLAML smoke test
python scripts/verify_env.py --quick  # skip the FLAML run
```

The script exits non-zero if anything is missing.

## Preprocessing

```bash
python src/data_preprocessing.py
```

Verifies the raw file, removes 12 duplicate rows, keeps `unknown` as its own category,
adds a `previously_contacted` flag (handles the `pdays = 999` sentinel), and maps `y` to 0/1.
Writes `data/cleaned_bank_data.csv` (41,176 × 22) and `report/data_preprocessing_summary.md`.

```python
from src.data import load_clean
df = load_clean()                    # for modelling: `duration` dropped
df_eda = load_clean(drop_leaky=False)  # for EDA: `duration` kept
```

## Exploratory analysis

```bash
python src/eda_visualization.py
```

Writes four 300-DPI figures to `figures/` and the findings write-up to `report/eda_summary.md`.

## AutoML training and evaluation

```bash
python src/train_and_evaluate.py              # 180 s FLAML search per model (~7 min total)
python src/train_and_evaluate.py --budget 30  # quick run
```

Trains a realistic model (`duration` dropped) and a leakage benchmark (`duration` kept) on the
same stratified 80/20 split, searching LightGBM, XGBoost, random forest and logistic regression
with 5-fold CV on ROC-AUC. Tunes a decision threshold on out-of-fold training predictions, then
evaluates on the test set. Writes `models/automl_*.pkl` (git-ignored; regenerate with the command
above), `models/model_metadata.json`, figures 05–08 and `report/automl_evaluation_summary.md`.

Load a saved model (from the project root, so `src.learners` can be imported):

```python
import joblib, json
from src.data import load_clean
model = joblib.load("models/automl_primary.pkl")
meta = json.load(open("models/model_metadata.json"))["primary"]
X = load_clean().drop(columns="y")[meta["features"]]
X = X.astype({c: "category" for c in X.select_dtypes(exclude="number").columns})
proba = model.predict_proba(X)[:, 1]
calls = proba >= meta["threshold"]
```

## Final report

`report/final_report.md` is the source; `report/final_report.docx` is the Word version
(US Letter, 11pt Calibri, single-spaced, cover page, figures sized to fit). Rebuild it after editing
the Markdown with:

```bash
node scripts/build_report_docx.js      # needs the docx npm package: npm install -g docx
```

## Run

```bash
jupyter lab                           # then create notebooks in notebooks/
```

Minimal usage from a notebook or script (run from the project root):

```python
from src.data import load_raw, split_xy, FIGURES_DIR, MODELS_DIR
from flaml import AutoML

df = load_raw("bank-additional-full")          # `duration` dropped, y encoded 0/1
X, y = split_xy(df)
X = X.astype({c: "category" for c in X.select_dtypes(exclude="number").columns})

automl = AutoML()
automl.fit(X, y, task="classification", metric="roc_auc", time_budget=300, seed=42)
```

From a notebook inside `notebooks/`, add the project root to the path first:
`import sys; sys.path.insert(0, "..")`.

Deactivate the environment with `deactivate`.
