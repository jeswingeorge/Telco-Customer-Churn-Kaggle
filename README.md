# Telco Customer Churn Prediction

Predict which telecom customers are likely to churn, so retention offers can be targeted.
Dataset: [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, ~26.5% churn).

## Context
"Predict behavior to retain customers. You can analyze all relevant customer data and develop focused customer retention programs." [IBM Sample Data Sets]

## Content
Each row represents a customer, each column contains customer’s attributes described on the column Metadata.

**The data set includes information about:**

- Customers who left within the last month – the column is called Churn
- Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
- Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
- Demographic info about customers – gender, age range, and if they have partners and dependents

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
