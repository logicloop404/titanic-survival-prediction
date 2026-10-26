# Data Directory

Place the official Kaggle Titanic dataset files here before running the training pipeline.

## Required Files

| File | Source |
|------|--------|
| `train.csv` | [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data) |
| `test.csv` | Same page |
| `gender_submission.csv` | Same page (optional) |

## Download Instructions

1. Go to https://www.kaggle.com/competitions/titanic/data
2. Sign in to your Kaggle account (free).
3. Accept the competition rules.
4. Click **Download All** (or download `train.csv` and `test.csv` individually).
5. Place those files inside this `data/` folder.

## After Placing Files

Run the training pipeline from the project root:

```bash
python train.py
```

This will preprocess the data, train models, save the best pipeline, and generate evaluation charts.
