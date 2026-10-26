"""
train.py
========
Entry-point script for the full training pipeline.

Run from the project root:
    python train.py

Steps
-----
1. Load train.csv from data/
2. Clean and engineer features; fit all imputers inside each training pipeline
3. Engineer features
4. Train four classifiers (LR, DT, RF, SVM)
5. Evaluate each model on the validation split
6. Select the best model by F1 score
7. Save the best pipeline to models/
8. Save evaluation charts to outputs/
9. Print a summary table
"""

import json
import pathlib
import sys
import pandas as pd

# Ensure the project root is on the Python path when running as a script
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.preprocessing import (
    load_dataset,
    drop_duplicates,
    get_missing_summary,
)
from src.features import engineer_features
from src.model_training import (
    train_all_models,
    select_best_model,
    save_pipeline,
    build_metrics_dataframe,
)
from src.evaluation import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_feature_importance,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_PATH = OUTPUT_DIR / "metrics.json"


def main():
    print("=" * 60)
    print("  Titanic Survival Prediction - Training Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    print("\n[1/7] Loading dataset ...")
    train_raw = load_dataset(DATA_DIR / "train.csv")
    print(f"  Loaded {len(train_raw)} rows x {len(train_raw.columns)} columns from train.csv")

    # ------------------------------------------------------------------
    # 2. Clean raw rows. Missing-value imputers are fitted inside each
    # scikit-learn pipeline using only that model's training split.
    # ------------------------------------------------------------------
    print("\n[2/7] Cleaning raw rows ...")
    train_raw = drop_duplicates(train_raw)
    missing_before = get_missing_summary(train_raw)
    if not missing_before.empty:
        print("  Missing values before imputation:")
        print(missing_before.to_string())

    # ------------------------------------------------------------------
    # 3. Feature engineering
    # ------------------------------------------------------------------
    print("\n[3/7] Engineering features ...")
    train_fe = engineer_features(train_raw)
    print(f"  Final feature set: {list(train_fe.columns)}")

    # ------------------------------------------------------------------
    # 4. Train models
    # ------------------------------------------------------------------
    print("\n[4/7] Training models ...")
    pipelines, metrics, X_train, y_train, X_val, y_val = train_all_models(train_fe)

    # ------------------------------------------------------------------
    # 5. Build metrics table
    # ------------------------------------------------------------------
    print("\n[5/7] Evaluating models ...")
    metrics_df = build_metrics_dataframe(metrics)
    print("\n" + metrics_df.to_string(index=False))

    # Save metrics to JSON for Streamlit app to load
    metrics_serialisable = {
        name: {k: v for k, v in m.items() if k != "confusion_matrix"}
        for name, m in metrics.items()
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_serialisable, f, indent=2)
    print(f"  Metrics saved -> {METRICS_PATH}")

    # Save metrics table as CSV
    metrics_df.to_csv(OUTPUT_DIR / "metrics_table.csv", index=False)

    # ------------------------------------------------------------------
    # 6. Save best model
    # ------------------------------------------------------------------
    print("\n[6/7] Selecting and saving best model ...")
    best_name = select_best_model(metrics)
    print(f"  Best model by F1 score: {best_name} (F1={metrics[best_name]['f1']:.4f})")
    best_pipeline = pipelines[best_name]
    saved_path = save_pipeline(best_pipeline, best_name)

    # Also save best model name so app.py can load it
    with open(OUTPUT_DIR / "best_model_name.txt", "w") as f:
        f.write(best_name)

    # ------------------------------------------------------------------
    # 7. Generate evaluation charts
    # ------------------------------------------------------------------
    print("\n[7/7] Generating evaluation charts ...")
    for name, pipe in pipelines.items():
        cm_path = plot_confusion_matrix(metrics[name], save=True)
        print(f"  Confusion matrix saved -> {cm_path}")

    comparison_path = plot_model_comparison(metrics_df, save=True)
    print(f"  Model comparison chart -> {comparison_path}")

    importance_path = plot_feature_importance(best_pipeline, [], save=True)
    if importance_path:
        print(f"  Feature importance chart -> {importance_path}")

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Model file : {saved_path}")
    print(f"  Charts     : {OUTPUT_DIR}/")
    print("  Run 'streamlit run app.py' to launch the web app.")
    print("=" * 60)


if __name__ == "__main__":
    main()
