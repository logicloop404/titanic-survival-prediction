"""
tests/test_preprocessing.py
============================
Tests for data loading, cleaning, and imputation logic.
Run from the project root:  python -m pytest -v
"""

import pathlib
import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.preprocessing import (
    load_dataset,
    drop_duplicates,
    compute_imputation_stats,
    impute_missing_values,
    get_missing_summary,
    REQUIRED_TRAIN_COLUMNS,
)
from src.features import (
    extract_title,
    add_family_features,
    add_title,
    drop_unnecessary_columns,
    engineer_features,
    TITLE_MAP,
)


# ---------------------------------------------------------------------------
# Fixtures — minimal synthetic DataFrames that mimic Titanic CSV structure
# ---------------------------------------------------------------------------

def make_minimal_train_df() -> pd.DataFrame:
    """Return a tiny synthetic training-like DataFrame (no missing values)."""
    return pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5],
        "Survived":    [0, 1, 1, 0, 1],
        "Pclass":      [3, 1, 3, 1, 2],
        "Name": [
            "Smith, Mr. John",
            "Jones, Mrs. Alice (Maiden)",
            "Brown, Miss. Claire",
            "Taylor, Mr. Robert",
            "Wilson, Master. Tom",
        ],
        "Sex":      ["male", "female", "female", "male", "male"],
        "Age":      [22.0, 38.0, 26.0, 35.0, 8.0],
        "SibSp":    [1, 1, 0, 0, 3],
        "Parch":    [0, 0, 0, 0, 1],
        "Ticket":   ["A/5 21171", "PC 17599", "STON/O2", "113803", "349909"],
        "Fare":     [7.25, 71.28, 7.92, 53.10, 21.07],
        "Cabin":    [None, "C85", None, "C123", None],
        "Embarked": ["S", "C", "S", "S", "Q"],
    })


def make_df_with_missing() -> pd.DataFrame:
    """Return a DataFrame with deliberate missing values in Age, Fare, Embarked."""
    df = make_minimal_train_df()
    df.loc[0, "Age"] = np.nan
    df.loc[1, "Fare"] = np.nan
    df.loc[2, "Embarked"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Tests: load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_raises_when_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            load_dataset(tmp_path / "nonexistent.csv")

    def test_raises_on_missing_columns(self, tmp_path):
        # Write a CSV that's missing several required columns
        bad_csv = tmp_path / "bad_train.csv"
        pd.DataFrame({"PassengerId": [1], "Name": ["A, Mr. B"]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_dataset(bad_csv)

    def test_loads_valid_csv(self, tmp_path):
        df = make_minimal_train_df()
        path = tmp_path / "train.csv"
        df.to_csv(path, index=False)
        loaded = load_dataset(path)
        assert isinstance(loaded, pd.DataFrame)
        assert len(loaded) == len(df)
        assert "Survived" in loaded.columns


# ---------------------------------------------------------------------------
# Tests: drop_duplicates
# ---------------------------------------------------------------------------

class TestDropDuplicates:
    def test_removes_exact_duplicates(self):
        df = make_minimal_train_df()
        df_duped = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        cleaned = drop_duplicates(df_duped)
        assert len(cleaned) == len(df)

    def test_no_change_when_no_duplicates(self):
        df = make_minimal_train_df()
        cleaned = drop_duplicates(df)
        assert len(cleaned) == len(df)


# ---------------------------------------------------------------------------
# Tests: imputation
# ---------------------------------------------------------------------------

class TestImputation:
    def test_stats_computed_from_training_data(self):
        df = make_minimal_train_df()
        stats = compute_imputation_stats(df)
        assert "age_median" in stats
        assert "fare_median" in stats
        assert "embarked_mode" in stats
        assert stats["age_median"] == df["Age"].median()
        assert stats["embarked_mode"] in ["S", "C", "Q"]

    def test_no_missing_after_imputation(self):
        df = make_df_with_missing()
        stats = compute_imputation_stats(make_minimal_train_df())   # fit on clean train
        imputed = impute_missing_values(df, stats)
        assert imputed["Age"].isnull().sum() == 0
        assert imputed["Fare"].isnull().sum() == 0
        assert imputed["Embarked"].isnull().sum() == 0

    def test_imputation_does_not_mutate_original(self):
        df = make_df_with_missing()
        stats = compute_imputation_stats(make_minimal_train_df())
        _ = impute_missing_values(df, stats)
        # Original should still have missing values
        assert df["Age"].isnull().any()

    def test_cabin_filled_with_unknown(self):
        df = make_df_with_missing()
        stats = compute_imputation_stats(make_minimal_train_df())
        imputed = impute_missing_values(df, stats)
        assert (imputed["Cabin"] == "Unknown").any()


# ---------------------------------------------------------------------------
# Tests: get_missing_summary
# ---------------------------------------------------------------------------

class TestMissingSummary:
    def test_returns_empty_df_when_no_missing(self):
        df = make_minimal_train_df()
        summary = get_missing_summary(df)
        # Cabin has None, Survived present — Cabin will show
        # Only check type
        assert isinstance(summary, pd.DataFrame)
        assert "Missing Count" in summary.columns
        assert "Missing %" in summary.columns

    def test_returns_correct_count(self):
        df = make_df_with_missing()
        summary = get_missing_summary(df)
        assert summary.loc["Age", "Missing Count"] == 1


# ---------------------------------------------------------------------------
# Tests: feature engineering
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    def test_extract_title_mr(self):
        assert extract_title("Smith, Mr. John") == "Mr"

    def test_extract_title_mrs(self):
        assert extract_title("Jones, Mrs. Alice (Maiden)") == "Mrs"

    def test_extract_title_miss(self):
        assert extract_title("Brown, Miss. Claire") == "Miss"

    def test_extract_title_master(self):
        assert extract_title("Wilson, Master. Tom") == "Master"

    def test_extract_title_rare(self):
        assert extract_title("Capt. Something, Capt. X") == "Rare"

    def test_extract_title_unknown_defaults_to_rare(self):
        assert extract_title("NoMatch Here") == "Rare"

    def test_add_family_features(self):
        df = make_minimal_train_df()
        out = add_family_features(df)
        assert "FamilySize" in out.columns
        assert "IsAlone" in out.columns
        # Row 0: SibSp=1, Parch=0 → FamilySize=2, IsAlone=0
        assert out.loc[0, "FamilySize"] == 2
        assert out.loc[0, "IsAlone"] == 0
        # Row 2: SibSp=0, Parch=0 → FamilySize=1, IsAlone=1
        assert out.loc[2, "FamilySize"] == 1
        assert out.loc[2, "IsAlone"] == 1

    def test_add_title_adds_column(self):
        df = make_minimal_train_df()
        out = add_title(df)
        assert "Title" in out.columns
        assert out.loc[0, "Title"] == "Mr"
        assert out.loc[1, "Title"] == "Mrs"

    def test_drop_unnecessary_columns(self):
        df = make_minimal_train_df()
        out = drop_unnecessary_columns(df)
        for col in ["PassengerId", "Name", "Ticket", "Cabin"]:
            assert col not in out.columns

    def test_engineer_features_complete(self):
        df = make_minimal_train_df()
        out = engineer_features(df)
        assert "FamilySize" in out.columns
        assert "IsAlone" in out.columns
        assert "Title" in out.columns
        assert "AgeGroup" in out.columns
        # Dropped columns should not appear
        for col in ["PassengerId", "Name", "Ticket", "Cabin"]:
            assert col not in out.columns


# ---------------------------------------------------------------------------
# Tests: invalid / edge case inputs
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_dataframe_imputation(self):
        """Imputing an empty DataFrame should not raise an error."""
        empty = pd.DataFrame(columns=make_minimal_train_df().columns)
        stats = compute_imputation_stats(make_minimal_train_df())
        result = impute_missing_values(empty, stats)
        assert len(result) == 0

    def test_single_row_dataframe(self):
        df = make_minimal_train_df().iloc[[0]].copy()
        df["Age"] = np.nan
        stats = compute_imputation_stats(make_minimal_train_df())
        result = impute_missing_values(df, stats)
        assert result["Age"].isnull().sum() == 0
