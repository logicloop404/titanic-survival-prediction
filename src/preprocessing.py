"""
preprocessing.py
================
Handles loading raw Titanic CSV files and applying basic cleaning:
- Validate required columns exist
- Drop duplicates
- Impute missing values for Age, Fare, and Embarked
- Convert data types

Design principle: All fit operations (e.g. computing medians) must be done
on TRAINING data only, then applied to validation/test data. This prevents
data leakage.
"""

import pathlib
import pandas as pd
import numpy as np

# Columns that MUST exist in the training file
REQUIRED_TRAIN_COLUMNS = {
    "PassengerId", "Survived", "Pclass", "Name", "Sex",
    "Age", "SibSp", "Parch", "Ticket", "Fare", "Embarked",
}

# Columns that MUST exist in the test file (no Survived)
REQUIRED_TEST_COLUMNS = REQUIRED_TRAIN_COLUMNS - {"Survived"}


def load_dataset(filepath: str | pathlib.Path) -> pd.DataFrame:
    """
    Load a CSV file and return a DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    filepath = pathlib.Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {filepath}\n"
            "Please download train.csv and test.csv from:\n"
            "  https://www.kaggle.com/competitions/titanic/data\n"
            "and place them in the data/ directory."
        )

    df = pd.read_csv(filepath)

    # Detect whether this is train or test based on presence of 'Survived'
    required = REQUIRED_TRAIN_COLUMNS if "Survived" in df.columns else REQUIRED_TEST_COLUMNS
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"The file '{filepath.name}' is missing required columns: {missing_cols}"
        )

    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and return cleaned DataFrame."""
    original_len = len(df)
    df = df.drop_duplicates()
    removed = original_len - len(df)
    if removed:
        print(f"[preprocessing] Removed {removed} duplicate row(s).")
    return df


def compute_imputation_stats(train_df: pd.DataFrame) -> dict:
    """
    Compute statistics needed for imputation from training data only.

    Returns a dict with:
      - age_median     : median Age from training data
      - fare_median    : median Fare from training data
      - embarked_mode  : most-common Embarked value from training data
    """
    return {
        "age_median": train_df["Age"].median(),
        "fare_median": train_df["Fare"].median(),
        "embarked_mode": train_df["Embarked"].mode()[0],
    }


def impute_missing_values(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Fill missing values using pre-computed statistics.

    Age     → median (robust to outliers)
    Fare    → median
    Embarked→ mode (most frequent port)
    Cabin   → 'Unknown' (too many missing to impute meaningfully)

    Parameters
    ----------
    df    : DataFrame to impute (may be train or test split)
    stats : dict returned by compute_imputation_stats (fit on train only)
    """
    df = df.copy()
    df["Age"] = df["Age"].fillna(stats["age_median"])
    df["Fare"] = df["Fare"].fillna(stats["fare_median"])
    df["Embarked"] = df["Embarked"].fillna(stats["embarked_mode"])
    if "Cabin" in df.columns:
        df["Cabin"] = df["Cabin"].fillna("Unknown")
    return df


def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary DataFrame showing missing value counts and percentages.
    Useful for EDA / notebook display.
    """
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(2)
    summary = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing %": missing_pct,
    })
    return summary[summary["Missing Count"] > 0].sort_values("Missing %", ascending=False)
