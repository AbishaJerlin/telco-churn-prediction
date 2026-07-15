# Telco Customer Churn Prediction — Decision Tree & Logistic Regression

Predicting telecom customer churn using a Decision Tree classifier, benchmarked against Logistic Regression, on the IBM Telco Customer Churn dataset. Built as my individual contribution to a 5-person university coursework project (Business Analytics with Data Visualisation, University of Surrey, 2025–26).

## About this repo

This was originally a **group project of 5 members**, each of whom independently implemented and tuned one classification algorithm on a shared, jointly-built preprocessed dataset, then compared results in a group dashboard.

**My role:** I implemented the **Decision Tree classifier** end-to-end — hyperparameter tuning, evaluation, feature importance analysis, calibration and the shared Logistic Regression baseline used across the whole group for comparison. That work is `telco_churn_decision_tree.ipynb`.

The preprocessing pipeline (`shared_preprocessing_pipeline.ipynb`) was co-developed by the whole team and is included here only because my notebook depends on its output (`telco_processed.csv`) to run — it is **not** presented as solely my work.

## Results (Decision Tree, test set)

| Metric | Score |
|---|---|
| Accuracy | 0.7484 |
| Precision | 0.5168 |
| Recall | 0.8235 |
| F1-score | 0.6351 |
| ROC-AUC | 0.8279 |
| PR-AUC | 0.5853 |
| MCC | 0.4870 |

The tuned Decision Tree achieved the highest F1-score among all five algorithms compared in the wider group project (Decision Tree, Naive Bayes, kNN, MLP, Linear SVM), with strong recall (82.4% of churners correctly identified) and good generalisation (overfit gap of only 0.014 between train and validation F1).

Feature importance and the Logistic Regression coefficients both pointed to **contract length, internet service type, and tenure** as the strongest churn drivers — customers without long-term contracts churn substantially more.

## Repository structure

```
├── telco_churn_decision_tree.ipynb        # My individual work: DT model, tuning, evaluation, LR benchmark
├── shared_preprocessing_pipeline.ipynb    # Group preprocessing notebook (dependency only)
├── telco-customer-churn.csv               # Raw dataset (source: Kaggle, see below)
├── telco_processed.csv                    # Cleaned/engineered dataset (output of preprocessing notebook)
├── requirements.txt
└── .gitignore
```

## Dataset

IBM Telco Customer Churn dataset — 7,043 customers, 21 attributes covering demographics, account info, subscribed services and billing. Target: whether the customer churned.

Source: [Kaggle — blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Methodology

- Missing/blank `TotalCharges` rows dropped (11 rows, ~0.16% of data, mostly zero-tenure customers)
- Five domain-informed engineered features: `TenureBucket`, `ServicesCount`, `AvgMonthlySpend`, `ChargesPerService`, `IsLongTermContract`
- Categorical variables one-hot encoded (`drop_first=True`); binary Yes/No fields mapped to 1/0
- Stratified 60/20/20 train/validation/test split, `random_state=42`
- Decision Tree trained on unscaled features; Logistic Regression on `StandardScaler`-scaled features (scaler fit on training data only, to avoid leakage)
- Hyperparameters (`max_depth`, `criterion`, `min_samples_split`) tuned via 5-fold stratified cross-validation, optimising F1-score
- Evaluated with Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC, and an overfit gap (train F1 − test F1)

## How to run

```bash
pip install -r requirements.txt
jupyter notebook
```

Run in order:
1. `shared_preprocessing_pipeline.ipynb` — produces `telco_processed.csv` from the raw data
2. `telco_churn_decision_tree.ipynb` — trains, tunes and evaluates the Decision Tree and Logistic Regression models

Reproducibility: `random_state=42` is used throughout for the split and all models.

## Acknowledgements

Preprocessing pipeline co-developed with my group project teammates as part of a University of Surrey coursework assignment. Their notebooks (Naive Bayes, kNN, MLP, Linear SVM) and the consolidated group dashboard are not included in this repository, as this repo showcases my individual contribution only.
