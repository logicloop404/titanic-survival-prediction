# Titanic Survival Prediction Using Machine Learning

> **A Final-Year Computer Engineering Mini Project**

---

## Abstract

This project builds a complete, end-to-end machine learning system to predict the survival of passengers aboard the RMS Titanic. It covers data preprocessing, exploratory data analysis, feature engineering, training and comparing four classification algorithms, evaluation, and deployment through an interactive Streamlit web application.

---

## Problem Statement

The sinking of the Titanic on 15 April 1912 resulted in the deaths of more than 1,500 people. This project investigates whether machine learning can identify patterns in passenger attributes (age, sex, class, fare, family size, port of embarkation) that reliably predict survival — demonstrating supervised classification on a real-world, widely studied dataset.

---

## Objectives

- Preprocess the raw Kaggle Titanic dataset, handling missing values and encoding categorical variables.
- Engineer meaningful features (FamilySize, IsAlone, Title).
- Train and compare Logistic Regression, Decision Tree, Random Forest, and SVM classifiers.
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrix.
- Select and save the best pipeline using scikit-learn and Joblib.
- Deploy an interactive prediction interface using Streamlit.

---

## Dataset

| Property | Value |
|----------|-------|
| Source   | [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data) |
| Train file | `train.csv` — 891 rows, 12 columns, includes `Survived` target |
| Test file  | `test.csv`  — 418 rows, 11 columns, no `Survived` column |
| Target variable | `Survived` (0 = did not survive, 1 = survived) |

### Key Features

| Column | Description |
|--------|-------------|
| `Pclass` | Passenger class (1 = Upper, 2 = Middle, 3 = Lower) |
| `Sex` | Passenger sex |
| `Age` | Age in years (~20% missing) |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Fare` | Ticket price in British pounds |
| `Embarked` | Port of embarkation (S = Southampton, C = Cherbourg, Q = Queenstown) |

---

## Technology Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core programming language |
| pandas, NumPy | Data manipulation |
| Matplotlib, Seaborn | Visualisation |
| scikit-learn | ML pipelines, preprocessing, classifiers |
| Joblib | Model serialisation |
| Streamlit | Interactive web application |
| Pytest | Unit testing |
| Jupyter Notebook | Exploratory data analysis |

---

## Project Structure

```
titanic-survival-prediction/
│
├── app.py                  ← Streamlit web application
├── train.py                ← Training pipeline entry-point
├── predict.py              ← CLI prediction utility
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── .gitignore
│
├── data/
│   ├── README.md           ← Dataset download instructions
│   └── .gitkeep
│
├── notebooks/
│   └── titanic_eda.ipynb   ← Full EDA and modelling notebook
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    ← Load, clean, impute
│   ├── features.py         ← Feature engineering
│   ├── model_training.py   ← Pipeline building, training, saving
│   └── evaluation.py       ← Metrics computation, chart generation
│
├── models/                 ← Saved pipeline files (.pkl)
├── outputs/                ← Evaluation charts and metrics JSON
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_prediction.py
│
└── docs/
    ├── project_report.md
    ├── viva_questions.md
    └── student_observations.md
```

---

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate (Windows PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

Activate (Linux / macOS):
```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Dataset Setup

1. Go to https://www.kaggle.com/competitions/titanic/data
2. Sign in and accept the competition rules.
3. Download `train.csv` and `test.csv`.
4. Place both files inside the `data/` directory.

---

## Running the Project

### Train all models

```bash
python train.py
```

This will:
- Load and preprocess `data/train.csv`
- Engineer features
- Train Logistic Regression, Decision Tree, Random Forest, and SVM
- Evaluate each model on a held-out validation split
- Save the best pipeline to `models/`
- Save evaluation charts to `outputs/`

### Launch the Streamlit app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Run the CLI predictor

```bash
python predict.py
```

Or with JSON input:
```bash
python predict.py --json "{\"Pclass\":1,\"Sex\":\"female\",\"Age\":29,\"SibSp\":0,\"Parch\":0,\"Fare\":100,\"Embarked\":\"S\"}"
```

### Run tests

```bash
python -m pytest -v
```

### Open the Jupyter notebook

```bash
jupyter notebook notebooks/titanic_eda.ipynb
```

---

## Screenshots

> *Run the app and add screenshots here.*

| Page | Screenshot |
|------|-----------|
| Home | *(placeholder)* |
| Exploratory Analysis | *(placeholder)* |
| Model Performance | *(placeholder)* |
| Survival Prediction | *(placeholder)* |

---

## Limitations and Future Improvements

| Limitation | Potential Improvement |
|------------|----------------------|
| ~20% of Age values are imputed | Use predictive imputation (e.g., regression based on Title and Pclass) |
| Cabin column dropped (>77% missing) | Extract deck letter as a feature |
| Small dataset (891 training examples) | Apply cross-validation for more robust evaluation |
| No hyperparameter tuning | Add GridSearchCV or RandomizedSearchCV |
| Binary classification only | Add probability calibration for better confidence estimates |

---

## Academic Disclaimer

Predictions are statistical estimates based on historical passenger data patterns. They do not reflect the actual fate of any individual.
