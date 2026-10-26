# Viva Questions and Answers
## Titanic Survival Prediction — Machine Learning Mini Project

> Study these 25 questions and answers before your viva. Be ready to explain every concept in your own words.

---

### Dataset and Problem

**Q1. What is the Titanic dataset and where did you get it?**  
The Titanic dataset is a publicly available benchmark dataset from the Kaggle competition "Titanic — Machine Learning from Disaster." It contains information about 891 passengers (training set) including demographic details, ticket information, and a binary survival label. We downloaded `train.csv` and `test.csv` from the Kaggle competition page.

---

**Q2. What type of machine learning problem is this?**  
This is a **supervised binary classification** problem. Supervised means the training data includes labelled examples (each passenger has a known `Survived` value of 0 or 1). Binary means there are exactly two outcome classes: survived or did not survive.

---

**Q3. Why did you use only `train.csv` for model development?**  
The `test.csv` file does not include the `Survived` column — it is the hidden evaluation set for the Kaggle competition leaderboard. We cannot evaluate model performance on it ourselves. Instead, we split `train.csv` into our own training and validation subsets (80/20) so that we can measure real metrics.

---

### Preprocessing and Missing Values

**Q4. What columns had missing values and how did you handle them?**  
- **Age** (~20% missing): filled with the median age from the training set. The median is preferred over the mean because it is robust to outliers (very old or very young passengers).
- **Cabin** (~77% missing): too sparse to impute meaningfully, so the column was dropped entirely.
- **Embarked** (2 rows missing): filled with the most frequent port (Southampton), which affected fewer than 0.3% of records.

---

**Q5. What is data leakage and how did you prevent it?**  
Data leakage occurs when information from the validation or test set "leaks" into the training process, making performance estimates unrealistically optimistic. We prevented it by computing imputation statistics (median, mode) and the scikit-learn `ColumnTransformer` fit operation **exclusively on training data**, then applying the fitted transformer to validation and test data without refitting.

---

**Q6. Why did you drop the `Name`, `Ticket`, and `Cabin` columns?**  
- `Name`: a raw string identifier. We extracted `Title` from it, which captures the useful information.
- `Ticket`: an alphanumeric code with no consistent pattern; no reliable predictive signal was found.
- `Cabin`: more than 77% of values are missing, making imputation unreliable. The remaining values would introduce highly sparse categories.
- `PassengerId`: a sequential identifier with no relationship to survival.

---

### Feature Engineering

**Q7. What is feature engineering? What new features did you create?**  
Feature engineering is the process of creating new input variables from existing ones to help the model better understand patterns. We created:
- `FamilySize = SibSp + Parch + 1` — total number of people in the passenger's family group including themselves.
- `IsAlone = 1 if FamilySize == 1` — binary flag for solo travellers.
- `Title` — extracted from the `Name` column (e.g. Mr, Mrs, Miss, Master, Rare). It encodes both gender and social status.

---

**Q8. Why is `Title` a useful feature?**  
`Title` compresses two pieces of information: gender (Mr vs Mrs/Miss) and social status (Dr, Rev, Col → grouped as Rare). Female titles (Mrs, Miss) are strongly correlated with higher survival due to the "women and children first" evacuation policy. Using `Title` alongside `Sex` makes the gender signal more nuanced.

---

**Q9. What are `SibSp` and `Parch`?**  
`SibSp` is the number of siblings (brothers, sisters) and spouses travelling with the passenger. `Parch` is the number of parents and children. Together they define the passenger's family group size, which we combined into `FamilySize`.

---

### Machine Learning Algorithms

**Q10. Briefly explain Logistic Regression.**  
Logistic Regression is a linear classification model. It computes a weighted sum of the input features, passes the result through a sigmoid (logistic) function to produce a probability between 0 and 1, then classifies the passenger as survived if this probability exceeds 0.5. The weights are learned by minimising log-loss (cross-entropy) on the training data. It is simple, fast, and highly interpretable.

---

**Q11. Briefly explain the Decision Tree Classifier.**  
A Decision Tree divides the feature space into rectangular regions by asking a series of binary questions (e.g. "Is Sex = female?"). At each node, the feature and threshold that produce the greatest reduction in Gini impurity are selected. The tree continues splitting until a maximum depth is reached or leaves are pure. It is very interpretable but prone to overfitting on training data.

---

**Q12. What is a Random Forest and why does it usually outperform a single Decision Tree?**  
A Random Forest is an ensemble of many decision trees. Each tree is trained on a **bootstrap sample** (random sample with replacement) of the training data, and at each split only a random subset of features is considered. Final prediction is the majority vote across all trees.

This reduces **variance** (overfitting) because the averaged prediction is smoother than any single tree. The decorrelation provided by random feature selection ensures the trees make different mistakes, so errors partially cancel out.

---

**Q13. What is an SVM and what does the RBF kernel do?**  
A Support Vector Machine finds the hyperplane that maximises the margin between the two classes. Points closest to the hyperplane are called support vectors. When data is not linearly separable, the **RBF (Radial Basis Function) kernel** implicitly maps the data to a higher-dimensional space where a linear boundary can separate the classes. In practice this allows SVMs to learn non-linear decision boundaries without explicitly computing the high-dimensional transformation.

---

### Model Evaluation

**Q14. What is a confusion matrix?**  
A confusion matrix is a 2×2 table showing the counts of:
- **True Positives (TP)**: predicted survived, actually survived
- **True Negatives (TN)**: predicted not survived, actually did not survive
- **False Positives (FP)**: predicted survived, actually did not survive
- **False Negatives (FN)**: predicted not survived, actually survived

It gives a detailed picture of the types of errors a model makes.

---

**Q15. Define accuracy, precision, recall, and F1-score.**  
- **Accuracy** = (TP + TN) / (TP + TN + FP + FN) — proportion of all correct predictions.
- **Precision** = TP / (TP + FP) — of all predicted survivors, how many actually survived?
- **Recall** = TP / (TP + FN) — of all actual survivors, how many did the model correctly identify?
- **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall) — harmonic mean; useful when classes are imbalanced.

---

**Q16. Why did you use F1-score to select the best model instead of accuracy?**  
The Titanic dataset is moderately imbalanced: about 62% of passengers did not survive and 38% survived. A model that always predicts "did not survive" would achieve ~62% accuracy while being completely useless. F1-score balances precision and recall, penalising models that fail to detect survivors (low recall) or that generate too many false alarms (low precision).

---

**Q17. What is overfitting? Did your models show any signs of it?**  
Overfitting occurs when a model memorises the training data instead of learning generalisable patterns. It performs very well on training data but poorly on unseen data. Signs include a large gap between training accuracy and validation accuracy. Decision Trees are especially prone to overfitting, which is why we limited `max_depth=5`. Random Forests mitigate this through ensemble averaging.

---

### Pipeline and Implementation

**Q18. What is a scikit-learn Pipeline?**  
A `Pipeline` chains multiple processing steps sequentially. In our case: `ColumnTransformer` (scales numeric features, one-hot encodes categorical features) → `Classifier`. The key benefit is that calling `pipeline.fit(X_train, y_train)` fits both the preprocessor and the classifier together, and calling `pipeline.predict(X_test)` applies the same transformations automatically — preventing data leakage and simplifying deployment.

---

**Q19. What is a ColumnTransformer?**  
A `ColumnTransformer` applies different transformers to different subsets of columns. In this project, numeric columns (`Age`, `Fare`, etc.) are standardised using `StandardScaler`, and categorical columns (`Sex`, `Embarked`, `Title`, etc.) are one-hot encoded using `OneHotEncoder`. This is cleaner and more reliable than applying transformations manually.

---

**Q20. Why did you use `StandardScaler` for numeric features?**  
`StandardScaler` transforms each numeric column to have zero mean and unit variance. This is important for distance-based and gradient-based algorithms (SVM, Logistic Regression) that are sensitive to feature scale. For example, `Fare` ranges from 0 to 500, while `SibSp` ranges from 0 to 8 — without scaling, Fare would dominate distance calculations. Tree-based models (Decision Tree, Random Forest) are scale-invariant but scaling does them no harm.

---

**Q21. What is `OneHotEncoding` and why is it used?**  
One-hot encoding converts a categorical variable into a set of binary (0/1) columns, one per category. For example, `Embarked` with values {S, C, Q} becomes three columns: `Embarked_S`, `Embarked_C`, `Embarked_Q`. Machine learning algorithms require numeric inputs; treating category codes as numbers would imply a false ordinal relationship (e.g. Q=2 > C=1 is meaningless).

---

**Q22. What is `joblib` and why was it used?**  
`joblib` is a Python library for efficient serialisation of large Python objects, particularly NumPy arrays. We used `joblib.dump()` to save the entire fitted pipeline (preprocessor + model) to a `.pkl` file, and `joblib.load()` to reload it for predictions in the Streamlit app. This means we never need to re-train the model when the app starts.

---

### Limitations and Broader Questions

**Q23. What are the main limitations of this project?**  
1. **Small dataset** — only 891 training examples; more data would improve model generalisation.
2. **Imputed Age** — ~20% of ages are estimated, introducing uncertainty.
3. **Dropped Cabin** — the passenger's deck location may have correlated with proximity to lifeboats.
4. **No hyperparameter tuning** — default or basic parameters were used; GridSearchCV could improve results.
5. **Binary outcome** — the model predicts aggregate patterns, not individual fates.

---

**Q24. Can this model predict whether a specific real person would have survived?**  
No. The model learns statistical correlations across 891 historical records. It produces a probability estimate based on similar passengers in the training data. It cannot account for individual actions, luck, or circumstances not captured in the data. The disclaimer in the app clearly states this limitation.

---

**Q25. If you had to improve this project, what would you do first?**  
The single most impactful improvement would be **hyperparameter tuning** using `GridSearchCV` or `RandomizedSearchCV` with cross-validation. This would systematically explore combinations of parameters (number of trees, max depth, learning rate) and select the combination with the best cross-validated F1-score — likely increasing performance by 1–3 percentage points while using the data more efficiently.

Other valuable improvements: predictive imputation for Age, extracting the Cabin deck letter, and adding SHAP explainability to the Streamlit app.
