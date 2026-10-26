# Student Observations Template
## Titanic Survival Prediction — Mini Project

> **Instructions:** Fill in each section after you have run the project on your own machine. Use real values from the output of `python train.py` and your own analysis of the results. Do not copy invented numbers.

---

## 1. Environment Setup

**Date of setup:** _______________  
**Python version:** _______________  
**Operating system:** _______________  
**Did the virtual environment create successfully?** _______________  
**Were there any installation errors?** _______________  

---

## 2. Dataset

**Date downloaded:** _______________  
**Source:** https://www.kaggle.com/competitions/titanic/data  
**Number of rows in train.csv:** _______________  
**Number of rows in test.csv:** _______________  
**Columns present:** _______________  

---

## 3. Missing Value Observations

After running the notebook or `train.py`, record the actual missing value counts:

| Column | Missing Count | Missing % | Strategy Used |
|--------|--------------|-----------|---------------|
| Age | | | |
| Cabin | | | |
| Embarked | | | |

**My observations about missing data:**  
*(Write 2–3 sentences in your own words. Why is Cabin so often missing? What does the Age imputation assume?)*

_______________

---

## 4. EDA Findings

**Survival rate in training data (%):** _______________  
**Female survival rate (%):** _______________  
**Male survival rate (%):** _______________  
**1st-class survival rate (%):** _______________  
**3rd-class survival rate (%):** _______________  

**Most surprising finding from EDA:**  
_______________

**Which visualisation was most informative and why:**  
_______________

---

## 5. Model Performance (fill after running train.py)

Copy the exact numbers from your terminal output:

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | | | | |
| Decision Tree | | | | |
| Random Forest | | | | |
| SVM | | | | |

---

## 6. Selected Model

**Model selected by the pipeline:** _______________  
**F1 score achieved:** _______________  
**Reason the code selected this model:**  
*(The code selects by highest F1 — confirm this matches the table above)*

_______________

**Do you agree with this choice? Why or why not:**  
_______________

---

## 7. Confusion Matrix Observations

For the best model, record the confusion matrix values:

|  | Predicted: Not Survived | Predicted: Survived |
|--|------------------------|---------------------|
| **Actual: Not Survived** | TN = | FP = |
| **Actual: Survived** | FN = | TP = |

**How many actual survivors did the model miss (False Negatives)?** _______________  
**Does this number concern you? Why?** _______________

---

## 8. Prediction App Experience

**Did the app launch successfully?** _______________  
**Which passenger profile did you test?** *(e.g. 1st-class female, 30 years old)*  
_______________  
**Prediction received:** _______________  
**Does the prediction match historical expectations? Why?** _______________  

---

## 9. Test Results

**Did all tests pass?** _______________  
**Number of tests passed / total:** _______________  
**Were any tests skipped? Why?** _______________  
**Any errors encountered:** _______________  

---

## 10. What I Would Improve

*(Write at least 3 genuine improvements you would make if you had more time)*

1. _______________
2. _______________
3. _______________

---

## 11. Personal Reflection

**What was the most difficult part of this project:**  
_______________

**What was the most interesting thing I learned:**  
_______________

**How would I explain this project to a non-technical audience:**  
_______________
