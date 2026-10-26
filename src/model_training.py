"""
model_training.py
=================
Builds scikit-learn Pipelines for four classifiers and returns fitted
pipeline objects together with their evaluation metrics.

Each pipeline consists of:
  1. ColumnTransformer  – scales numeric columns, one-hot encodes categoricals
  2. Classifier         – one of LR, DT, RF, SVM

The entire pipeline (preprocessor + model) is saved with joblib so that
the same transformations are applied automatically at prediction time.
"""

import pathlib
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from src.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, ALL_FEATURES
from src.evaluation import evaluate_model

RANDOM_STATE = 42
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


def build_preprocessor() -> ColumnTransformer:
    """
    Create a ColumnTransformer that:
      - Standardises numeric features (zero mean, unit variance)
      - One-hot encodes categorical features (drop first to avoid multicollinearity)
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return preprocessor


def get_model_definitions() -> dict:
    """Return a dict of {model_name: classifier_instance}."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=RANDOM_STATE
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE
        ),
    }


def build_pipeline(classifier) -> Pipeline:
    """Wrap a preprocessor and classifier into a single sklearn Pipeline."""
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", classifier),
    ])


def train_all_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[dict, dict, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split data, train all models, evaluate, and return results.

    Parameters
    ----------
    df        : Feature-engineered training DataFrame (must contain TARGET column)
    test_size : Fraction of data held out for validation

    Returns
    -------
    pipelines   : dict of {name: fitted Pipeline}
    metrics     : dict of {name: metrics_dict}
    X_train, y_train, X_val, y_val : splits used for training / evaluation
    """
    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    model_defs = get_model_definitions()
    pipelines: dict = {}
    metrics: dict = {}

    for name, clf in model_defs.items():
        print(f"  Training: {name} ...")
        pipe = build_pipeline(clf)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        y_prob = pipe.predict_proba(X_val)[:, 1] if hasattr(pipe, "predict_proba") else None

        metrics[name] = evaluate_model(y_val, y_pred, name)
        pipelines[name] = pipe

    return pipelines, metrics, X_train, y_train, X_val, y_val


def select_best_model(metrics: dict) -> str:
    """
    Select the model with the highest F1 score on the validation set.
    F1 is preferred over accuracy because the Titanic dataset is moderately
    imbalanced (about 38 % survived) and F1 balances precision with recall.
    """
    best = max(metrics, key=lambda name: metrics[name]["f1"])
    return best


def save_pipeline(pipeline: Pipeline, model_name: str) -> pathlib.Path:
    """
    Save a fitted pipeline to disk using joblib.

    Returns the path where the file was saved.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.lower().replace(" ", "_")
    filepath = MODEL_DIR / f"titanic_{safe_name}_pipeline.pkl"
    joblib.dump(pipeline, filepath)
    print(f"  Saved pipeline -> {filepath}")
    return filepath


def load_pipeline(filepath: str | pathlib.Path) -> Pipeline:
    """
    Load a previously saved pipeline from disk.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist. Instruct the user to run train.py.
    """
    filepath = pathlib.Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Model file not found: {filepath}\n"
            "Please run 'python train.py' first to train and save the model."
        )
    return joblib.load(filepath)


def build_metrics_dataframe(metrics: dict) -> pd.DataFrame:
    """Convert the nested metrics dict into a tidy DataFrame for display."""
    rows = []
    for name, m in metrics.items():
        rows.append({
            "Model": name,
            "Accuracy": round(m["accuracy"], 4),
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1 Score": round(m["f1"], 4),
        })
    return pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
