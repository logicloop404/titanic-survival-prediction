"""
predict.py
==========
Command-line prediction utility.

Usage
-----
Predict for a single passenger (interactive or via flags):
    python predict.py

Or pass JSON directly:
    python predict.py --json '{"Pclass":1,"Sex":"female","Age":29,"SibSp":0,"Parch":0,"Fare":100,"Embarked":"S"}'

The script loads the best saved pipeline and prints the prediction.
"""

import argparse
import json
import math
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.preprocessing import compute_imputation_stats
from src.features import engineer_features, ALL_FEATURES
from src.model_training import load_pipeline

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

REQUIRED_INPUTS = ("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")


def validate_passenger(passenger: dict) -> dict:
    """Validate and normalise one passenger record before feature engineering."""
    if not isinstance(passenger, dict):
        raise ValueError("Passenger input must be a JSON object or dictionary.")

    missing = [key for key in REQUIRED_INPUTS if key not in passenger]
    if missing:
        raise ValueError(f"Missing passenger input field(s): {', '.join(missing)}")

    try:
        normalised = {
            "Pclass": int(passenger["Pclass"]),
            "Sex": str(passenger["Sex"]).strip().lower(),
            "Age": float(passenger["Age"]),
            "SibSp": int(passenger["SibSp"]),
            "Parch": int(passenger["Parch"]),
            "Fare": float(passenger["Fare"]),
            "Embarked": str(passenger["Embarked"]).strip().upper(),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("Passenger fields must contain valid numeric and text values.") from exc

    if normalised["Pclass"] not in {1, 2, 3}:
        raise ValueError("Pclass must be 1, 2, or 3.")
    if normalised["Sex"] not in {"male", "female"}:
        raise ValueError("Sex must be 'male' or 'female'.")
    if normalised["Embarked"] not in {"S", "C", "Q"}:
        raise ValueError("Embarked must be one of S, C, or Q.")
    if normalised["Age"] < 0 or not math.isfinite(normalised["Age"]):
        raise ValueError("Age must be a finite non-negative number.")
    if normalised["SibSp"] < 0 or normalised["Parch"] < 0:
        raise ValueError("SibSp and Parch must be non-negative integers.")
    if normalised["Fare"] < 0 or not math.isfinite(normalised["Fare"]):
        raise ValueError("Fare must be a finite non-negative number.")

    return normalised


def find_best_pipeline() -> pathlib.Path:
    """Locate the saved best-model pipeline file."""
    best_name_file = OUTPUT_DIR / "best_model_name.txt"
    if best_name_file.exists():
        name = best_name_file.read_text().strip()
        safe = name.lower().replace(" ", "_")
        candidate = MODELS_DIR / f"titanic_{safe}_pipeline.pkl"
        if candidate.exists():
            return candidate

    # Fallback: any .pkl in models/
    pkls = list(MODELS_DIR.glob("*.pkl"))
    if pkls:
        return pkls[0]

    raise FileNotFoundError(
        "No trained model found in models/.\n"
        "Please run 'python train.py' first."
    )


def build_input_dataframe(passenger: dict) -> pd.DataFrame:
    """
    Convert a passenger dict into a DataFrame ready for pipeline inference.
    Applies the same feature engineering used during training.

    The passenger dict must contain:
      Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

    A fake Name is added to allow title extraction.
    """
    passenger = validate_passenger(passenger)
    sex = passenger["Sex"]
    title_placeholder = "Mr." if sex == "male" else "Mrs."
    row = {
        "PassengerId": 0,
        "Name": f"Unknown, {title_placeholder} Passenger",
        "Ticket": "X",
        "Cabin": None,
        "Pclass": int(passenger["Pclass"]),
        "Sex": sex,
        "Age": float(passenger["Age"]),
        "SibSp": int(passenger["SibSp"]),
        "Parch": int(passenger["Parch"]),
        "Fare": float(passenger["Fare"]),
        "Embarked": passenger["Embarked"],
    }
    df = pd.DataFrame([row])
    df = engineer_features(df)
    # Keep only the model features
    for col in ALL_FEATURES:
        if col not in df.columns:
            df[col] = 0   # safe fallback
    return df[ALL_FEATURES]


def predict(passenger: dict) -> dict:
    """
    Run inference for a single passenger.

    Returns
    -------
    dict with keys: prediction (int), label (str), probability (float)
    """
    pipeline_path = find_best_pipeline()
    pipeline = load_pipeline(pipeline_path)

    input_df = build_input_dataframe(passenger)
    prediction = int(pipeline.predict(input_df)[0])
    label = "Survived" if prediction == 1 else "Did Not Survive"

    probability = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(input_df)[0]
        probability = round(float(proba[prediction]), 4)

    return {
        "prediction": prediction,
        "label": label,
        "probability": probability,
        "model_used": pipeline_path.stem,
    }


def interactive_input() -> dict:
    """Prompt the user to enter passenger details from the terminal."""
    print("\n  Enter passenger details (press Enter to use defaults):\n")
    pclass = int(input("  Passenger Class [1/2/3] (default 3): ").strip() or 3)
    sex = input("  Sex [male/female] (default male): ").strip() or "male"
    age = float(input("  Age (default 28): ").strip() or 28)
    sibsp = int(input("  Siblings/Spouses aboard (default 0): ").strip() or 0)
    parch = int(input("  Parents/Children aboard (default 0): ").strip() or 0)
    fare = float(input("  Fare paid (default 15): ").strip() or 15)
    embarked = input("  Embarkation port [S/C/Q] (default S): ").strip() or "S"

    return {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
    }


def main():
    parser = argparse.ArgumentParser(description="Titanic Survival Predictor")
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help='Passenger data as a JSON string, e.g. \'{"Pclass":1,...}\'',
    )
    args = parser.parse_args()

    if args.json:
        passenger = json.loads(args.json)
    else:
        passenger = interactive_input()

    print("\n  Running prediction ...")
    result = predict(passenger)

    print("\n" + "=" * 45)
    print(f"  Prediction : {result['label'].upper()}")
    if result["probability"] is not None:
        pct = result["probability"] * 100
        print(f"  Confidence : {pct:.1f} %")
    print(f"  Model used : {result['model_used']}")
    print("=" * 45)
    print("\n  Note: This is a statistical estimate based on historical\n"
          "  Titanic passenger data. It does not reflect certainty\n"
          "  about any individual's actual fate.\n")


if __name__ == "__main__":
    main()
