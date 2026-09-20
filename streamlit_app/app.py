"""
Customer churn risk predictor (Streamlit demo).

Run locally:   streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

DATA_PATH = Path(__file__).parent / "telco_clean.csv"
TARGET, ID_COL = "Churn", "customerID"
KEY_COLS = ["Contract", "tenure", "MonthlyCharges", "InternetService",
            "OnlineSecurity", "TechSupport", "PaymentMethod"]
INTERNET_ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies"]


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def train_model():
    df = load_data()
    y = df[TARGET]
    X = df.drop(columns=[TARGET, ID_COL])
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    clf = HistGradientBoostingClassifier(
        learning_rate=0.06, max_depth=3, max_iter=120, max_leaf_nodes=8,
        min_samples_leaf=58, l2_regularization=9.0, random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
    baseline = {c: (X[c].median() if c in num else X[c].mode()[0]) for c in X.columns}
    return pipe, auc, baseline, list(X.columns)


def with_total(row):
    row = dict(row)
    row["TotalCharges"] = row["tenure"] * row["MonthlyCharges"]
    return row


def explain(pipe, row, baseline, columns):
    """Change in churn risk when each feature is set to a typical customer's value."""
    base_p = pipe.predict_proba(pd.DataFrame([row], columns=columns))[:, 1][0]
    variants, names = [], []
    for col in columns:
        if col == "TotalCharges":
            continue
        alt = dict(row)
        alt[col] = baseline[col]
        if col in ("tenure", "MonthlyCharges"):
            alt = with_total(alt)
        variants.append(alt)
        names.append(col)
    p_alt = pipe.predict_proba(pd.DataFrame(variants, columns=columns))[:, 1]
    effects = pd.Series(base_p - p_alt, index=names)
    return effects.reindex(effects.abs().sort_values(ascending=False).index)


df = load_data()
pipe, auc, baseline, columns = train_model()


def default_index(col, opts):
    """Start each dropdown at the most typical value."""
    return opts.index(baseline[col]) if baseline[col] in opts else 0


def choices(col, opts):
    return opts, default_index(col, opts)

st.title("📉 Customer Churn Risk Predictor")
st.caption("Enter a customer's details to see their predicted churn risk, the main "
           "reasons, and whether a retention offer is worth it.")

with st.sidebar:
    st.header("Business assumptions")
    cost = st.slider("Cost of a retention offer ($)", 5, 200, 50, 5)
    success = st.slider("Offer success rate", 0.05, 0.9, 0.30, 0.05)
    horizon = st.slider("Months of revenue saved", 3, 36, 12)
    margin = st.slider("Profit margin", 0.1, 1.0, 0.5, 0.05)
    st.divider()
    st.caption(f"Model: gradient boosting trained on {len(df):,} telecom customers "
               f"(IBM Telco dataset). Held-out ROC-AUC: **{auc:.2f}**.")
    st.caption("Dollar figures depend on the assumptions above, which are illustrative.")

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Customer details")
    inputs = {}
    internet = st.selectbox("InternetService", *choices("InternetService", sorted(df["InternetService"].unique())))
    inputs["InternetService"] = internet

    def field(col, container):
        series = df[col]
        if col == "tenure":
            return container.slider("Tenure (months)", int(series.min()), int(series.max()), 12)
        if col == "MonthlyCharges":
            return container.slider("Monthly charges ($)", float(series.min()),
                                    float(series.max()), 70.0, 0.5)
        if series.dtype.kind in "if":
            opts = sorted(series.dropna().unique().tolist())
            return container.selectbox(col, opts, index=default_index(col, opts),
                                       format_func=lambda v: "Yes" if v == 1 else "No")
        opts = sorted(series.unique())
        if col in INTERNET_ADDONS:
            opts = ["No internet service"] if internet == "No" else \
                [o for o in opts if o != "No internet service"]
        if col == "MultipleLines" and inputs.get("PhoneService") == "No":
            opts = ["No phone service"]
        elif col == "MultipleLines":
            opts = [o for o in opts if o != "No phone service"]
        return container.selectbox(col, opts, index=default_index(col, opts))

    for col in [c for c in KEY_COLS if c != "InternetService"]:
        inputs[col] = field(col, st)

    with st.expander("More details (optional)"):
        inputs["PhoneService"] = st.selectbox("PhoneService", *choices(
            "PhoneService", sorted(df["PhoneService"].unique())))
        for col in columns:
            if col in inputs or col == "TotalCharges":
                continue
            inputs[col] = field(col, st)

row = with_total({c: inputs.get(c, baseline[c]) for c in columns})

with right:
    st.subheader("Prediction")
    p = pipe.predict_proba(pd.DataFrame([row], columns=columns))[:, 1][0]
    band = "🟢 Low risk" if p < 0.2 else "🟡 Medium risk" if p < 0.5 else "🔴 High risk"
    c1, c2 = st.columns(2)
    c1.metric("Churn probability", f"{p:.0%}")
    c2.metric("Risk level", band)
    st.progress(float(p))

    value = row["MonthlyCharges"] * horizon * margin
    expected = p * success * value
    st.markdown("**Retention decision**")
    if expected >= cost:
        st.success(f"Contact this customer. Expected saved value \\${expected:,.0f} "
                   f"vs. offer cost \\${cost:,.0f} (net about \\${expected - cost:,.0f}).")
    else:
        st.info(f"Don't offer a discount. Expected saved value \\${expected:,.0f} "
                f"is below the \\${cost:,.0f} offer cost.")
    st.caption("Rule: contact if churn probability x offer success rate x value of the "
               "customer is at least the offer cost.")

    st.markdown("**Main reasons (vs. a typical customer)**")
    effects = explain(pipe, row, baseline, columns).head(5)
    for feat, eff in effects.items():
        if abs(eff) < 0.005:
            continue
        arrow = "🔺" if eff > 0 else "🔻"
        st.markdown(f"{arrow} **{feat}** = {row[feat]}: "
                    f"{'raises' if eff > 0 else 'lowers'} risk by about {abs(eff) * 100:.0f} points")
    st.caption("Approximate: each factor is swapped for a typical customer's value and the "
               "change in predicted risk is shown.")
