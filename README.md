![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4c72b0)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

# Car Insurance Claim Prediction — Single-Feature Logistic Regression

A beginner logistic regression project that identifies which customer attribute best predicts whether someone will file a car insurance claim.

## Overview

Given a dataset of ~10,000 policyholders, this project builds a separate logistic regression model for each available feature (excluding customer ID), evaluates each model's accuracy against the actual claim outcome, and identifies the single strongest predictor.

## Dataset

`car_insurance.csv` — 10,000 rows, 18 columns, including:

- **Target**: `outcome` — whether the customer filed a claim (1) or not (0)
- **Demographic**: `age`, `gender`, `education`, `income`, `married`, `children`
- **Driving-related**: `driving_experience`, `speeding_violations`, `duis`, `past_accidents`
- **Vehicle-related**: `vehicle_ownership`, `vehicle_year`, `vehicle_type`, `annual_mileage`
- **Other**: `credit_score`, `postal_code`

Two columns (`credit_score`, `annual_mileage`) contained missing values.

## Methodology

1. **Cleaning** — filled missing numeric values (`credit_score`, `annual_mileage`) with the column median.
2. **Encoding** — converted categorical string columns (`driving_experience`, `education`, `income`, `vehicle_year`, `vehicle_type`) to numeric codes with `pd.Categorical(...).codes`.
3. **Single-feature modeling** — for every remaining feature, fit `outcome ~ feature` using `statsmodels.formula.api.logit`, then computed a confusion matrix (`.pred_table()`) and accuracy score.
4. **Selection** — compared accuracy across all 16 features to find the best individual predictor.
5. **Baseline check** — compared each feature's accuracy against the majority-class baseline (68.7%, since ~69% of customers never claim) to separate genuinely informative features from ones that just default to the majority prediction.
6. **Visualization** — plotted a logistic `regplot` of the winning feature against claim outcome to visually confirm the relationship.

## Results

| Feature | Accuracy |
|---|---|
| **driving_experience** | **0.7771** |
| age | 0.7747 |
| vehicle_ownership | 0.7351 |
| credit_score | 0.7053 |
| annual_mileage | 0.6904 |
| all other features | 0.6867 (= baseline) |

**Best single predictor: `driving_experience`**, with 77.71% accuracy.

Predicted claim probability drops sharply across experience buckets — from roughly 63% for the least experienced drivers to near 0% for the most experienced — confirming a strong relationship rather than a spurious accuracy score.

## Key insight

Accuracy alone can be misleading on imbalanced targets. Roughly 69% of customers never file a claim, so a model that ignores the data and always predicts "no claim" already scores 68.7%. Several features in this dataset landed at exactly that number, meaning they added no real predictive signal — only `driving_experience`, `age`, `vehicle_ownership`, `credit_score`, and `annual_mileage` beat the baseline meaningfully.

## Tech stack

- Python
- pandas — data cleaning and encoding
- statsmodels — logistic regression (`logit`)
- seaborn / matplotlib — visualization

## Project structure

```
.
├── car_insurance.csv
├── notebook.ipynb (or .py)
└── README.md
```

## Possible next steps

- Build a multivariate logistic regression combining the informative features
- Evaluate with train/test split instead of in-sample accuracy
- Add precision, recall, and ROC-AUC given the class imbalance
- Test interaction effects (e.g. driving experience × age)

