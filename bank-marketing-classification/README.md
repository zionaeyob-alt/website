# Bank Marketing – Term Deposit Classification

Predict whether a client subscribes to a term deposit (`y`) from the
[UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
(Moro et al., 2014).

## Layout

```
bank-marketing-classification/
├── data/
│   ├── raw/            # original UCI files (committed, read-only)
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
