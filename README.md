# Telco Customer Churn Prediction

Predict which telecom customers are likely to churn, so retention offers can be targeted.
Dataset: [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, ~26.5% churn).

## Results
_To be filled in after model selection._

## Live app
_To be filled in after deploying to Cloud Run._

## Project organization
```
├── data/{raw,interim,processed,external}   <- git-ignored; raw CSV goes in data/raw/
├── models/          <- trained pipeline + metadata
├── notebooks/       <- N.0-jg-<topic>.ipynb
├── references/      <- data dictionary and other reference material
├── reports/figures/ <- generated plots
├── churn/           <- source package (config, dataset, features, evaluate, plots, modeling/)
├── app/             <- Streamlit app
├── tests/
└── docs/
```

## How to run
```bash
uv sync
```
_More commands to be added as the pipeline is built._
