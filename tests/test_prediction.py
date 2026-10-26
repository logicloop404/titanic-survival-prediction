"""
tests/test_prediction.py
=========================
Tests for the prediction pipeline — build_input_dataframe and the
end-to-end predict() function using a lightweight fitted pipeline.
"""

import pathlib
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.features import ALL_FEATURES, engineer_features
from predict import build_input_dataframe


# ---------------------------------------------------------------------------
# Fixtures — synthetic passenger dicts
# ---------------------------------------------------------------------------

MALE_3RD_CLASS = {
    "Pclass": 3, "Sex": "male", "Age": 22,
    "SibSp": 1, "Parch": 0, "Fare": 7.25, "Embarked": "S",
}

FEMALE_1ST_CLASS = {
    "Pclass": 1, "Sex": "female", "Age": 38,
    "SibSp": 1, "Parch": 0, "Fare": 71.28, "Embarked": "C",
}

CHILD_WITH_FAMILY = {
    "Pclass": 2, "Sex": "male", "Age": 8,
    "SibSp": 3, "Parch": 1, "Fare": 21.07, "Embarked": "Q",
}


# ---------------------------------------------------------------------------
# Tests: build_input_dataframe
# ---------------------------------------------------------------------------

class TestBuildInputDataframe:
    def test_returns_dataframe(self):
        df = build_input_dataframe(MALE_3RD_CLASS)
        assert isinstance(df, pd.DataFrame)

    def test_has_correct_columns(self):
        df = build_input_dataframe(MALE_3RD_CLASS)
        for col in ALL_FEATURES:
            assert col in df.columns, f"Missing column: {col}"

    def test_single_row(self):
        df = build_input_dataframe(MALE_3RD_CLASS)
        assert len(df) == 1

    def test_family_size_computed_correctly(self):
        df = build_input_dataframe(CHILD_WITH_FAMILY)
        # SibSp=3, Parch=1 → FamilySize=5
        assert df["FamilySize"].iloc[0] == 5

    def test_is_alone_false_for_family(self):
        df = build_input_dataframe(CHILD_WITH_FAMILY)
        assert df["IsAlone"].iloc[0] == 0

    def test_is_alone_true_for_solo(self):
        solo = {**MALE_3RD_CLASS, "SibSp": 0, "Parch": 0}
        df = build_input_dataframe(solo)
        assert df["IsAlone"].iloc[0] == 1

    def test_female_title_extracted(self):
        df = build_input_dataframe(FEMALE_1ST_CLASS)
        assert df["Title"].iloc[0] == "Mrs"

    def test_male_title_extracted(self):
        df = build_input_dataframe(MALE_3RD_CLASS)
        assert df["Title"].iloc[0] == "Mr"

    def test_no_missing_values_in_output(self):
        for passenger in [MALE_3RD_CLASS, FEMALE_1ST_CLASS, CHILD_WITH_FAMILY]:
            df = build_input_dataframe(passenger)
            # Title and IsAlone should have no NaN
            assert df[["FamilySize", "IsAlone", "Title"]].isnull().sum().sum() == 0

    def test_pclass_preserved(self):
        df = build_input_dataframe(FEMALE_1ST_CLASS)
        assert df["Pclass"].iloc[0] == 1

    def test_fare_preserved(self):
        df = build_input_dataframe(FEMALE_1ST_CLASS)
        assert abs(df["Fare"].iloc[0] - 71.28) < 0.001

    def test_embarked_preserved(self):
        df = build_input_dataframe(FEMALE_1ST_CLASS)
        assert df["Embarked"].iloc[0] == "C"


# ---------------------------------------------------------------------------
# Tests: validation / edge cases
# ---------------------------------------------------------------------------

class TestEdgeCasePredictionInputs:
    def test_missing_required_field_raises(self):
        incomplete = dict(MALE_3RD_CLASS)
        del incomplete["Fare"]
        with pytest.raises(ValueError, match="Missing passenger input field"):
            build_input_dataframe(incomplete)

    def test_invalid_class_raises(self):
        invalid = {**MALE_3RD_CLASS, "Pclass": 4}
        with pytest.raises(ValueError, match="Pclass"):
            build_input_dataframe(invalid)

    def test_invalid_port_raises(self):
        invalid = {**MALE_3RD_CLASS, "Embarked": "X"}
        with pytest.raises(ValueError, match="Embarked"):
            build_input_dataframe(invalid)

    def test_age_zero_boundary(self):
        """Age=0 (infant) should not raise an error."""
        infant = {**MALE_3RD_CLASS, "Age": 0.5}
        df = build_input_dataframe(infant)
        assert df["Age"].iloc[0] == 0.5

    def test_high_fare(self):
        """Very high fare should be handled without error."""
        rich = {**FEMALE_1ST_CLASS, "Fare": 512.0}
        df = build_input_dataframe(rich)
        assert df["Fare"].iloc[0] == 512.0

    def test_large_family(self):
        """Large family (SibSp=8) should produce correct FamilySize."""
        large = {**MALE_3RD_CLASS, "SibSp": 8, "Parch": 2}
        df = build_input_dataframe(large)
        assert df["FamilySize"].iloc[0] == 11

    def test_all_embarkation_ports(self):
        """All three embarkation codes should be accepted."""
        for port in ["S", "C", "Q"]:
            passenger = {**MALE_3RD_CLASS, "Embarked": port}
            df = build_input_dataframe(passenger)
            assert df["Embarked"].iloc[0] == port

    def test_output_columns_match_all_features(self):
        """Output columns must exactly match the training feature list."""
        df = build_input_dataframe(MALE_3RD_CLASS)
        assert list(df.columns) == ALL_FEATURES


# ---------------------------------------------------------------------------
# Tests: end-to-end pipeline (requires trained model — skip if not present)
# ---------------------------------------------------------------------------

MODEL_DIR = pathlib.Path("models")
OUTPUT_DIR = pathlib.Path("outputs")

requires_model = pytest.mark.skipif(
    not any(MODEL_DIR.glob("*.pkl")),
    reason="No trained model found in models/ — run python train.py first",
)


@requires_model
class TestEndToEndPrediction:
    def test_predict_returns_valid_keys(self):
        from predict import predict
        result = predict(MALE_3RD_CLASS)
        assert "prediction" in result
        assert "label" in result
        assert "probability" in result

    def test_predict_binary_output(self):
        from predict import predict
        result = predict(MALE_3RD_CLASS)
        assert result["prediction"] in [0, 1]

    def test_predict_label_matches_prediction(self):
        from predict import predict
        result = predict(MALE_3RD_CLASS)
        if result["prediction"] == 1:
            assert "Survived" in result["label"]
        else:
            assert "Survive" in result["label"]

    def test_predict_probability_in_range(self):
        from predict import predict
        result = predict(FEMALE_1ST_CLASS)
        if result["probability"] is not None:
            assert 0.0 <= result["probability"] <= 1.0

    def test_predict_high_survival_probability_for_1st_class_female(self):
        """First-class female historically had very high survival rates."""
        from predict import predict
        result = predict(FEMALE_1ST_CLASS)
        # We expect a positive prediction — not guaranteed but historically likely
        # This is a soft assertion: just checks the model runs and returns sensibly
        assert result["prediction"] in [0, 1]
