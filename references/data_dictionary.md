# Data dictionary: Telco Customer Churn

**Source:** IBM sample data, via Kaggle [`blastchar/telco-customer-churn`](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
**Raw file:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`: 7,043 rows (one per customer), 21 columns, no missing values except 11 blank `TotalCharges`.
**Snapshot:** each row is a customer at one point in time. `Churn` = left within the last month.

**Model use** key:
- **Target**: what the model predicts.
- **Feature**: used by the model.
- **Dropped**: kept in the cleaned data but not used by the model. The reasons are also in `churn/config.py → DROP_COLS`, and each drop is to be confirmed with cross-validation in the baseline phase.

## Target

| Column | Type | Values (count) | Description | Model use |
|---|---|---|---|---|
| `Churn` | Yes/No | No (5,174), Yes (1,869) | Customer left within the last month. 26.5% churn, an imbalanced target, so accuracy is not used as the main metric. | **Target**, mapped to 1 = Yes, 0 = No |

## Identifier

| Column | Type | Values | Description | Model use |
|---|---|---|---|---|
| `customerID` | String | 7,043 unique values, e.g. `7590-VHVEG` | Customer identifier. | **Dropped**: no predictive value. Kept in the data to label batch predictions. |

## Demographics

| Column | Type | Values (count) | Description | Model use |
|---|---|---|---|---|
| `gender` | Categorical | Male (3,555), Female (3,488) | Customer's gender. | **Dropped**: no relationship with churn (both about 26%; Cramér's V = 0.000, p = 0.47). |
| `SeniorCitizen` | Binary, stored as **0/1** integer | 0 (5,901), 1 (1,142) | Whether the customer is 65 or older. The only binary column stored as a number rather than Yes/No. | **Feature** (categorical, not numeric). Seniors churn at 41.7% vs 23.6%. |
| `Partner` | Yes/No | No (3,641), Yes (3,402) | Has a partner. | **Feature**: 33.0% churn without vs 19.7% with. |
| `Dependents` | Yes/No | No (4,933), Yes (2,110) | Has dependents (children, etc.). | **Feature**: 31.3% vs 15.5%. Adds signal beyond `Partner`. |

## Account information

| Column | Type | Values / range | Description | Model use |
|---|---|---|---|---|
| `tenure` | Integer (months) | 0–72; median 29 | Months the customer has been with the company. U-shaped distribution; 72 is the dataset's cap. | **Feature**: churn 52.9% in the first 6 months vs 9.5% after 4 years. |
| `Contract` | Categorical | Month-to-month (3,875), Two year (1,695), One year (1,473) | Contract term. | **Feature**: strongest single driver (42.7% / 11.3% / 2.8%). |
| `PaperlessBilling` | Yes/No | Yes (4,171), No (2,872) | Receives bills electronically. | **Feature**: 33.6% vs 16.3%; holds within every payment method. |
| `PaymentMethod` | Categorical | Electronic check (2,365), Mailed check (1,612), Bank transfer (automatic) (1,544), Credit card (automatic) (1,522) | How the customer pays. | **Feature**: electronic check 45.3% vs 15–19% for the others. All 4 levels kept. |
| `MonthlyCharges` | Float (\$) | 18.25–118.75; median 70.35 | Current monthly bill. | **Feature** |
| `TotalCharges` | Float (\$), **stored as text** in the raw file | 0–8,684.80 after cleaning | Total billed to date. **Quirk:** 11 values are blank `" "`; all have `tenure == 0` (not yet billed), so they are filled with **0**. | **Dropped**: ≈ `tenure × MonthlyCharges`, so collinear with them. |

## Services

| Column | Type | Values (count) | Description | Model use |
|---|---|---|---|---|
| `PhoneService` | Yes/No | Yes (6,361), No (682) | Has phone service. | **Dropped**: fully contained in `MultipleLines` (its "No phone service" level). |
| `MultipleLines` | Categorical | No (3,390), Yes (2,971), No phone service (682) | Has more than one phone line. "No phone service" = `PhoneService == "No"`. | **Feature**: weak signal (25–29%). |
| `InternetService` | Categorical | Fiber optic (3,096), DSL (2,421), No (1,526) | Internet type. | **Feature**: fibre 41.9%, DSL 19.0%, none 7.4%. The only column that records "no internet". |
| `OnlineSecurity` | Yes/No/No internet service | No (3,498), Yes (2,019), No internet service (1,526) | Online security add-on. | **Feature**\*: 41.8% without vs 14.6% with (internet customers). |
| `OnlineBackup` | Yes/No/No internet service | No (3,088), Yes (2,429), No internet service (1,526) | Online backup add-on. | **Feature**\* |
| `DeviceProtection` | Yes/No/No internet service | No (3,095), Yes (2,422), No internet service (1,526) | Device protection add-on. | **Feature**\* |
| `TechSupport` | Yes/No/No internet service | No (3,473), Yes (2,044), No internet service (1,526) | Tech support add-on. | **Feature**\*: 41.6% without vs 15.2% with. |
| `StreamingTV` | Yes/No/No internet service | No (2,810), Yes (2,707), No internet service (1,526) | Streams TV. | **Feature**\*: weak signal, under review in CV. |
| `StreamingMovies` | Yes/No/No internet service | No (2,785), Yes (2,732), No internet service (1,526) | Streams movies. | **Dropped**: no churn signal beyond `StreamingTV`. |

\* **"No internet service"** in these add-on columns (`config.NO_INTERNET_COLS`) is identical to `InternetService == "No"` (1,526 rows in every column). Inside the model pipeline, `churn.features.collapse_no_internet` replaces it with "No", so the fact is recorded only once, in `InternetService`.

## Data quality checks

- **Missing values:** none, apart from the 11 blank `TotalCharges` (handled above).
- **Duplicates:** no duplicate rows. 22 rows are identical once `customerID` is removed; they are kept, because they are different customers who happen to have the same profile.
- **Value ranges:** all categorical levels are as listed above; no negative or out-of-range numbers.
