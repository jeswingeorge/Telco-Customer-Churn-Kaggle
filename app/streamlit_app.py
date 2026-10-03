"""Streamlit app: score one customer (sidebar form) or a whole CSV, using the saved model.

Run from the repo root:
  uv run streamlit run app/streamlit_app.py
then open http://localhost:8501 in a browser.

How Streamlit works, in one sentence: every time you touch a widget, Streamlit re-runs
this whole file from top to bottom (like "Run All" in a notebook) with the new widget
values, and redraws the page. So `x = st.slider(...)` simply gives you the current value.

All scoring goes through churn.modeling.predict, the same code the CLI uses, and the
saved pipeline does all the feature engineering. This file has no feature logic of its own.
"""

import pandas as pd
import streamlit as st

from churn import config
from churn.modeling import predict as churn_predict

HIGH_RISK = churn_predict.HIGH_RISK
TENURE_MIN, TENURE_MAX = churn_predict.TENURE_MIN, churn_predict.TENURE_MAX

# Must be the first Streamlit command: browser-tab title and a wide layout.
st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="wide")


# ---------------------------------------------------------------------------
# 1. Load the model once
# ---------------------------------------------------------------------------
# Because the script re-runs on every click, loading the 1-2 MB model each time would be
# wasteful. @st.cache_resource runs the function once, then hands back the same object on
# every later re-run (and to every user of the app).
@st.cache_resource
def load_model():
    return churn_predict.load_model()


model, metadata = load_model()
THRESHOLD = metadata["threshold"]

# The categories the fitted OneHotEncoder learned, e.g. {"Contract": ["Month-to-month", ...]}.
# Reading them from the model (not typing them here) means the form can never offer a value
# the model has never seen.
_encoder = model.named_steps["prep"].named_transformers_["cat"]
CATEGORIES = {col: list(levels) for col, levels in zip(_encoder.feature_names_in_, _encoder.categories_)}


# ---------------------------------------------------------------------------
# 2. "Why this score?" helper (stand-in for SHAP, which is deferred)
# ---------------------------------------------------------------------------
def top_reasons(customer: dict, n: int = 5) -> pd.DataFrame:
    """Each feature's push on the churn log-odds for this customer.

    Logistic regression: log-odds = intercept + sum(coef * transformed value). So
    coef * transformed value is exactly how much each column moves this customer's score.
    The one-hot columns of one feature (e.g. the 3 Contract columns) are summed back into
    that feature. Numbers are scaled, so tenure's push is relative to the average customer.
    It's a rough guide to the model's reasoning, not a causal claim.
    """
    X = pd.DataFrame([customer])
    raw = model[:2].transform(X)        # after collapse + add_features (has tenure_group)
    Z = model[:-1].transform(X)         # everything except the classifier: scaled + one-hot
    coefs = pd.Series(model[-1].coef_[0], index=Z.columns)
    contrib = Z.iloc[0] * coefs

    def original_feature(column_name):  # "cat__Contract_One year" -> "Contract"
        name = column_name.split("__", 1)[1]
        # Longest names first: "tenure_group_0-6" also starts with "tenure_", so checking
        # "tenure" first would wrongly put the tenure_group columns under tenure.
        for feature in sorted(config.NUM_COLS + config.CAT_COLS, key=len, reverse=True):
            if name == feature or name.startswith(feature + "_"):
                return feature
        return name

    by_feature = contrib.groupby(original_feature).sum()
    by_feature = by_feature[by_feature.abs() >= 0.01]  # elastic-net set some coefs to 0: no effect
    top =by_feature.reindex(by_feature.abs().sort_values(ascending=False).index).head(n)
    return pd.DataFrame({
        "Feature": top.index,
        "Customer's value": [str(raw.iloc[0][f]) for f in top.index],
        "Effect": ["⬆ raises risk" if v > 0 else "⬇ lowers risk" for v in top.values],
        "Push on log-odds": top.values.round(2),
    })


# ---------------------------------------------------------------------------
# 3. Sidebar: one customer's details
# ---------------------------------------------------------------------------
# Anything written as st.sidebar.<widget> appears in the left panel.
st.sidebar.header("Customer details")

# A dict of {column name: value}, the format predict_one() expects.
customer = {}

st.sidebar.subheader("Account")
customer["tenure"] = st.sidebar.slider(
    "tenure (months with the company)", min_value=TENURE_MIN, max_value=TENURE_MAX, value=12,
    help="Limited to 0-72 months, the range the model was trained on.",
)
customer["Contract"] = st.sidebar.selectbox("Contract", CATEGORIES["Contract"])
customer["PaymentMethod"] = st.sidebar.selectbox("PaymentMethod", CATEGORIES["PaymentMethod"])
customer["PaperlessBilling"] = st.sidebar.selectbox("PaperlessBilling", CATEGORIES["PaperlessBilling"])
customer["MonthlyCharges"] = st.sidebar.number_input(
    "MonthlyCharges ($)", min_value=0.0, max_value=200.0, value=70.0, step=5.0,
)

st.sidebar.subheader("Demographics")
customer["SeniorCitizen"] = st.sidebar.selectbox("SeniorCitizen", CATEGORIES["SeniorCitizen"])
customer["Partner"] = st.sidebar.selectbox("Partner", CATEGORIES["Partner"])
customer["Dependents"] = st.sidebar.selectbox("Dependents", CATEGORIES["Dependents"])

st.sidebar.subheader("Services")
customer["MultipleLines"] = st.sidebar.selectbox("MultipleLines", CATEGORIES["MultipleLines"])
customer["InternetService"] = st.sidebar.selectbox("InternetService", CATEGORIES["InternetService"])

# Add-ons only exist with internet. Since the script re-runs on every change, we can simply
# check the value chosen just above and grey these out (and force "No") when there's none.
no_internet = customer["InternetService"] == "No"
for col in config.NO_INTERNET_COLS:
    customer[col] = st.sidebar.selectbox(col, CATEGORIES[col], disabled=no_internet)
    if no_internet:
        customer[col] = "No"


# ---------------------------------------------------------------------------
# 4. Main page: three tabs
# ---------------------------------------------------------------------------
st.title("📉 Telco Customer Churn Predictor")
st.caption(
    f"{metadata['model']} · flags a customer as high risk when P(churn) ≥ {THRESHOLD:.3f}"
)

tab_single, tab_batch, tab_about = st.tabs(["Single customer", "Batch scoring (CSV)", "About the model"])

# ---- Tab 1: the customer from the sidebar ----
# Code inside "with tab_x:" is drawn inside that tab.
with tab_single:
    result = churn_predict.predict_one(customer, model, metadata)
    proba = result["churn_probability"]

    left, right = st.columns(2)  # two side-by-side boxes
    with left:
        st.metric("Churn probability", f"{proba:.1%}")
        st.progress(proba)  # a bar filled to proba (0-1)
    with right:
        if result["risk_label"] == HIGH_RISK:
            st.error(f"**{HIGH_RISK}**: worth a retention offer.")
        else:
            st.success(f"**{result['risk_label']}**: no action needed.")
        st.caption(
            f"Threshold {THRESHOLD:.3f} = the best precision while still catching ≥ 75% of "
            "churners (chosen on out-of-fold predictions). It is below 0.5 on purpose: "
            "missing a churner costs more than an unneeded offer."
        )

    st.subheader("Why this score?")
    st.dataframe(top_reasons(customer), hide_index=True, width="stretch")
    st.caption(
        "Up to 5 features by their push on this customer's log-odds (coefficient × scaled/one-hot "
        "value of the logistic regression). Positive = towards churn."
    )

# ---- Tab 2: score a whole CSV ----
with tab_batch:
    st.write(
        "Upload a CSV with these columns (extra columns such as `customerID` are fine and "
        "are kept in the output). The raw Kaggle file works as-is."
    )
    st.code(", ".join(metadata["input_columns"]), language=None)

    # A one-row template built from the sidebar customer, so users see the expected format.
    template = pd.DataFrame([customer])[metadata["input_columns"]].to_csv(index=False)
    st.download_button("Download a template CSV", template, file_name="churn_template.csv")

    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is not None:  # None until the user uploads something
        df = pd.read_csv(uploaded)
        try:
            scores = churn_predict.predict(df, model, metadata)
        except ValueError as err:  # e.g. a required column is missing
            st.error(str(err))
            st.stop()  # stop this run here; nothing below is drawn

        scored = pd.concat([df, scores], axis=1)
        n_ok = int((scores["problem"] == "").sum())
        n_high = int((scores["risk_label"] == HIGH_RISK).sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows scored", f"{n_ok:,} of {len(df):,}")
        c2.metric("High risk", f"{n_high:,}", f"{n_high / max(n_ok, 1):.1%} of scored", delta_color="off")
        c3.metric("Not scored", f"{len(df) - n_ok:,}")
        if n_ok < len(df):
            st.warning("Some rows were not scored; the `problem` column says why "
                       "(e.g. tenure outside 0-72, or an unknown category).")

        # Highest risk first, so the retention team sees who to call at the top.
        st.dataframe(scored.sort_values("churn_probability", ascending=False), hide_index=True)
        st.download_button("Download scored CSV", scored.to_csv(index=False),
                           file_name="churn_scores.csv", mime="text/csv")

# ---- Tab 3: model card from metadata.json ----
with tab_about:
    st.subheader("Model card")
    st.markdown(
        f"""
- **Model:** {metadata['model']}, `C = {metadata['params']['C']:.3f}`, `l1_ratio = {metadata['params']['l1_ratio']:.3f}`
- **Decision threshold:** {THRESHOLD:.3f} ({metadata['threshold_rule']})
- **Data:** Kaggle Telco Customer Churn, {metadata['n_train']:,} training / {metadata['n_test']:,} test customers (stratified 80/20 split), trained {metadata['trained_on']}
- **Features:** {", ".join(metadata['model_features'])}
- **Why logistic regression:** tuned XGBoost was only +0.001 ROC-AUC better; LR is calibrated, fast and explainable.
- **Limits:** tenure must be 0-72 months; categories must match the training data; reasons are associations, not causes.
"""
    )
    st.subheader("Metrics")
    metrics = pd.DataFrame({
        "Cross-validation (out-of-fold, train)": metadata["cv_oof_metrics"],
        "Test set (held out, used once)": metadata["test_metrics"],
    })
    st.dataframe(metrics, width="stretch")
    st.caption("Precision/recall/F1 at the saved threshold. Accuracy is for reference only "
               "(about 26.5% of customers churn).")
