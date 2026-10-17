"""Check that the environment and data are ready.

Usage: python scripts/verify_env.py [--quick]
  --quick  skip the FLAML smoke test
"""
import importlib
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PACKAGES = ["pandas", "numpy", "sklearn", "matplotlib", "seaborn",
            "ucimlrepo", "requests", "flaml", "lightgbm", "xgboost", "joblib"]
EXPECTED_ROWS = {"bank-additional-full": 41188, "bank-additional": 4119,
                 "bank-full": 45211, "bank": 4521}


def main() -> int:
    ok = True
    print(f"Python {sys.version.split()[0]} ({sys.executable})")

    print("\nPackages:")
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"  [ok]   {name} {getattr(mod, '__version__', '')}")
        except ImportError as exc:
            print(f"  [FAIL] {name}: {exc}")
            ok = False

    from src.data import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, load_raw, split_xy

    print("\nData:")
    for name, rows in EXPECTED_ROWS.items():
        try:
            df = load_raw(name, drop_leaky=False)
            status = "ok" if len(df) == rows else "FAIL"
            ok &= status == "ok"
            print(f"  [{status:<4}] {name}: {len(df):,} rows x {df.shape[1]} cols, "
                  f"positive rate {df['y'].mean():.1%}")
        except FileNotFoundError as exc:
            print(f"  [FAIL] {name}: {exc}")
            ok = False

    for d in (PROCESSED_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if ok and "--quick" not in sys.argv:
        print("\nFLAML smoke test (10s on bank-additional sample):")
        # sklearn 1.9 warns on every RandomForest fit inside FLAML; harmless.
        warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
        from flaml import AutoML
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        X, y = split_xy(load_raw("bank-additional"))
        X = X.astype({c: "category" for c in X.select_dtypes(exclude="number").columns})
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        automl = AutoML()
        automl.fit(X_tr, y_tr, task="classification", metric="roc_auc",
                   time_budget=10, estimator_list=["lgbm", "xgboost", "rf"],
                   seed=42, verbose=0)
        auc = roc_auc_score(y_te, automl.predict_proba(X_te)[:, 1])
        print(f"  [ok]   best learner: {automl.best_estimator}, holdout ROC-AUC {auc:.3f}")

    print("\nEnvironment OK" if ok else "\nEnvironment has problems (see FAIL lines)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
