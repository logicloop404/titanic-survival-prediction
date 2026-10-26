# Project Report: Titanic Survival Prediction Using Machine Learning

**Course:** Computer Engineering — Mini Project  
**Topic:** Supervised Classification with scikit-learn  
**Dataset:** Kaggle Titanic Competition

---

## Abstract

This report documents the design, implementation, and evaluation of a machine learning system that predicts whether a Titanic passenger survived the 1912 disaster. Four classification algorithms — Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine — were trained on the official Kaggle dataset and evaluated using accuracy, precision, recall, and F1-score. The best-performing model was saved and deployed through an interactive Streamlit web application. All reported metrics are computed from actual model predictions on a held-out validation split.

---

## 1. Introduction

Machine learning classification enables computers to learn patterns from labelled data and apply those patterns to new examples. The Titanic dataset is a classic supervised learning benchmark because it combines numerical features (age, fare), categorical features (sex, class, port), and engineering opportunities (title extraction, family size) in a human-understandable context.

---

## 2. Problem Definition

**Goal:** Given passenger attributes (class, sex, age, family size, fare, embarkation port), predict whether that passenger survived (`Survived = 1`) or did not survive (`Survived = 0`).

**Type:** Binary classification.  
**Evaluation measure:** F1-score (balances precision and recall; the exact class distribution is calculated from the downloaded training file).

---

## 3. Proposed Methodology

```
Raw CSV → Cleaning → Imputation → Feature Engineering → Train/Val Split
   → ColumnTransformer (scale + encode) → Classifier → Evaluate → Save
```

A scikit-learn **Pipeline** combines preprocessing and the classifier so that the same transformations are automatically applied at inference time, preventing data leakage.

---

## 4. System Architecture

| Layer | Component | Technology |
|-------|-----------|-----------|
| Data I/O | CSV loading, validation | pandas |
| Preprocessing | Imputation, cleaning | pandas, NumPy |
| Feature Engineering | Title, FamilySize, IsAlone | pandas |
| ML Pipeline | ColumnTransformer + Classifier | scikit-learn |
| Serialisation | Save/load fitted pipeline | Joblib |
| Web Application | 4-page interactive dashboard | Streamlit |
| Testing | Unit tests | Pytest |
| Notebooks | Exploratory analysis | Jupyter |

---

## 5. Dataset and Preprocessing

### 5.1 Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Training samples | Displayed by the notebook/app from `train.csv` |
| Test samples | Displayed by the notebook from `test.csv` |
| Features (raw) | Displayed by the notebook/app |
| Target | Survived (binary) |
| Survival rate (train) | Calculated from `train.csv` |

### 5.2 Missing Values

| Column | Missing Count | Strategy |
|--------|--------------|----------|
| Age | Calculated from `train.csv` | Median imputation fitted inside each training pipeline |
| Cabin | Calculated from `train.csv` | Dropped when not used as a model feature |
| Embarked | Calculated from `train.csv` | Most-frequent imputation fitted inside each training pipeline |

The `ColumnTransformer` fits all imputation statistics only on each model's training fold. The validation fold is transformed using those fitted statistics, preventing information leakage.

### 5.3 Feature Engineering

| Feature | Source | Rationale |
|---------|--------|-----------|
| `FamilySize` | `SibSp + Parch + 1` | Small families survived at higher rates |
| `IsAlone` | `FamilySize == 1` | Solo travel correlated with lower survival |
| `Title` | Extracted from `Name` | Encodes gender and social status |

Rare titles (Dr, Rev, Capt, etc.) are grouped into a single `Rare` category to reduce sparse categories that would inflate the one-hot encoded feature space.

### 5.4 Columns Dropped

| Column | Reason |
|--------|--------|
| `PassengerId` | Identifier — no predictive signal |
| `Name` | Raw string — Title already extracted |
| `Ticket` | Alphanumeric code — no reliable pattern |
| `Cabin` | >77% missing — insufficient for imputation |

---

## 6. Algorithms Used

### 6.1 Logistic Regression
Models survival probability using the sigmoid function of a linear combination of features. Highly interpretable via feature coefficients. Assumes a linear decision boundary.

### 6.2 Decision Tree
Recursively splits data on the feature that maximally reduces Gini impurity. Depth is limited (`max_depth=5`) to reduce overfitting. Very interpretable as a visual tree diagram.

### 6.3 Random Forest
An ensemble of 200 decision trees, each trained on a bootstrap sample with random feature subsets. Predictions are made by majority vote. Reduces overfitting compared to a single tree while capturing non-linear interactions.

### 6.4 Support Vector Machine (SVM)
Finds the hyperplane that maximises the margin between survival classes. Uses an RBF (radial basis function) kernel to model non-linear boundaries. Requires feature scaling (provided by the pipeline preprocessor).

---

## 7. Experimental Results

> **Important:** This section intentionally does not contain invented metric values. Run `python train.py` with the official Kaggle file to generate `outputs/metrics_table.csv`, `outputs/metrics.json`, and the charts used by the application.

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | Generated at runtime | Generated at runtime | Generated at runtime | Generated at runtime |
| Decision Tree | Generated at runtime | Generated at runtime | Generated at runtime | Generated at runtime |
| Random Forest | Generated at runtime | Generated at runtime | Generated at runtime | Generated at runtime |
| SVM | Generated at runtime | Generated at runtime | Generated at runtime | Generated at runtime |

The preferred model is selected by the highest measured validation F1-score and recorded in `outputs/best_model_name.txt`; no model is assumed to win before training.

---

## 8. Conclusion

- The application and notebook calculate survival patterns from the supplied dataset.
- Engineered features (Title, FamilySize, IsAlone) compress information from raw columns into model-ready signals.
- The selected classifier is the one with the highest measured validation F1-score for that run.
- The scikit-learn Pipeline ensures consistent preprocessing at both training and prediction time.

---

## 9. Future Scope

- **Hyperparameter optimisation** using GridSearchCV or Optuna.
- **Advanced feature engineering**: deck letter from Cabin, fare-per-person normalised by family size.
- **Ensemble / stacking**: combine predictions from multiple classifiers.
- **Explainability**: integrate SHAP values to explain individual predictions.
- **Deployment**: containerise with Docker and deploy to a cloud service.

---

## 10. References

1. Kaggle. *Titanic — Machine Learning from Disaster.* https://www.kaggle.com/competitions/titanic
2. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
3. McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of SciPy 2010*.
4. Streamlit Inc. *Streamlit Documentation.* https://docs.streamlit.io
