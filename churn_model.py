import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_demo_data(n=8000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": np.arange(n),
        "tenure_months": rng.integers(1, 72, n),
        "plan": rng.choice(["basic", "standard", "premium"], n, p=[.5, .35, .15]),
        "contract": rng.choice(["monthly", "annual"], n, p=[.7, .3]),
        "monthly_spend": rng.normal(60, 20, n).clip(10),
        "logins_last_30d": rng.poisson(12, n),
        "support_tickets_90d": rng.poisson(1.2, n),
        "late_payments_12m": rng.poisson(0.4, n),
    })
    logit = (
        -1.0
        - 0.04 * df.tenure_months
        + 1.2 * (df.contract == "monthly")
        - 0.10 * df.logins_last_30d
        + 0.35 * df.support_tickets_90d
        + 0.50 * df.late_payments_12m
        + 0.01 * (df.monthly_spend - 60)
        + 1.0
    )
    df["churned"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def build_preprocessor(X):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def top_decile_lift(y_true, scores):
    df = pd.DataFrame({"y": np.asarray(y_true), "s": scores}).sort_values("s", ascending=False)
    top = df.head(max(1, len(df) // 10))
    return top.y.mean() / df.y.mean()


def evaluate(name, model, X_test, y_test):
    p = model.predict_proba(X_test)[:, 1]
    print(f"\n=== {name} ===")
    print(f"ROC-AUC:            {roc_auc_score(y_test, p):.3f}")
    print(f"PR-AUC:             {average_precision_score(y_test, p):.3f}  "
          f"(baseline = churn rate {y_test.mean():.3f})")
    print(f"Top-decile lift:    {top_decile_lift(y_test, p):.2f}x")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Path to your customer data")
    ap.add_argument("--target", default="churned")
    ap.add_argument("--id", default="customer_id")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    df = pd.read_csv(args.csv) if args.csv else make_demo_data()
    print(f"Loaded {len(df):,} rows | churn rate: {df[args.target].mean():.1%}")

    y = df[args.target]
    X = df.drop(columns=[c for c in [args.target, args.id] if c in df.columns])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    logit = Pipeline([("pre", build_preprocessor(X)),
                      ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    logit.fit(X_train, y_train)
    evaluate("Logistic regression (baseline)", logit, X_test, y_test)

    gb = Pipeline([("pre", build_preprocessor(X)),
                   ("clf", HistGradientBoostingClassifier(
                       max_depth=4, learning_rate=0.06, max_iter=300,
                       class_weight="balanced", random_state=42))])
    gb.fit(X_train, y_train)
    p = evaluate("Gradient boosting", gb, X_test, y_test)

    print(f"\nClassification report at threshold {args.threshold}:")
    print(classification_report(y_test, (p >= args.threshold).astype(int), digits=3))

    imp = permutation_importance(gb, X_test, y_test, scoring="roc_auc",
                                 n_repeats=5, random_state=42)
    ranked = pd.Series(imp.importances_mean, index=X_test.columns).sort_values(ascending=False)
    print("Feature importance (drop in AUC when shuffled):")
    print(ranked.round(4).to_string())

    out = df.copy()
    out["churn_probability"] = gb.predict_proba(X)[:, 1]
    out.sort_values("churn_probability", ascending=False).to_csv("churn_scores.csv", index=False)
    print("\nSaved churn_scores.csv (all customers ranked by churn risk)")


if __name__ == "__main__":
    main()
