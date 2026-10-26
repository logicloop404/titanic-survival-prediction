"""
features.py
===========
Feature engineering for the Titanic dataset.

New features created:
  FamilySize  = SibSp + Parch + 1  (total people including the passenger)
  IsAlone     = 1 if FamilySize == 1, else 0
  Title       = extracted from passenger Name (e.g. Mr, Mrs, Miss, Master)
  AgeGroup    = binned age category for visualisation (not used in model directly)

Columns dropped before modelling:
  Name, Ticket, Cabin, PassengerId
  (identifiers / too sparse / already encoded via Title)
"""

import re
import pandas as pd
import numpy as np

# Map rare titles to broader groups so the model generalises better
TITLE_MAP = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    # Rare / honorific titles → "Rare"
    "Dr": "Rare",
    "Rev": "Rare",
    "Col": "Rare",
    "Major": "Rare",
    "Mlle": "Miss",   # French equivalent of Miss
    "Countess": "Rare",
    "Ms": "Miss",
    "Lady": "Rare",
    "Jonkheer": "Rare",
    "Don": "Rare",
    "Dona": "Rare",
    "Capt": "Rare",
    "Sir": "Rare",
    "Mme": "Mrs",     # French equivalent of Mrs
}

AGE_BINS = [0, 12, 18, 35, 60, 120]
AGE_LABELS = ["Child", "Teenager", "YoungAdult", "Adult", "Senior"]


def extract_title(name: str) -> str:
    """
    Pull the title from a passenger name string.
    Names are in the format: 'Last, Title. First'
    Returns the mapped title group (Mr, Mrs, Miss, Master, Rare).
    """
    match = re.search(r",\s*([^\.]+)\.", name)
    if match:
        raw_title = match.group(1).strip()
        return TITLE_MAP.get(raw_title, "Rare")
    return "Rare"


def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add FamilySize and IsAlone columns."""
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def add_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and map title from the Name column."""
    df = df.copy()
    df["Title"] = df["Name"].apply(extract_title)
    return df


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin Age into categorical groups.
    Used for visualisation in EDA; not passed to the model directly.
    """
    df = df.copy()
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=True,
    )
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns not used as model features:
      - PassengerId : identifier, no predictive signal
      - Name        : raw string; Title already extracted
      - Ticket      : alphanumeric code with no reliable pattern
      - Cabin        : >77 % missing in training data
    """
    cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    existing = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=existing)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the complete feature engineering pipeline in the correct order.
    Call this AFTER imputing missing values.
    """
    df = add_title(df)
    df = add_family_features(df)
    df = add_age_group(df)
    df = drop_unnecessary_columns(df)
    return df


# ---------------------------------------------------------------------------
# Column lists used by the scikit-learn ColumnTransformer in model_training.py
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "IsAlone"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Survived"
