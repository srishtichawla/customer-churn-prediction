"""
Probability calibration: can we trust the model's scores as real probabilities?

Usage:
    python churn_calibration.py --csv telco_clean.csv --target Churn --id customerID
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BLUE, RED, GREEN, GRAY = "#4878a8", "#d95f4b", "#4a9a6b", "#8c8c8c"


def make_pipeline(X):
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
    return Pipeline([("pre", pre), ("clf", clf)])


def expected_calibration_error(y, p, bins=10):
    """Average gap between predicted and actual churn rate, weighted by bin size."""
    idx = np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1])
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="telco_clean.csv")
    ap.add_argument("--target", default="Churn")
    ap.add_argument("--id", default="customerID")
    ap.add_argument("--outdir", default="charts")
    ap.add_argument("--cost", type=float, default=50.0)
    ap.add_argument("--success", type=float, default=0.3)
    ap.add_argument("--horizon", type=float, default=12.0)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--value-col", default="MonthlyCharges")
    ap.add_argument("--flat-value", type=float, default=400.0)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    y = df[args.target]
    X = df.drop(columns=[c for c in [args.target, args.id] if c in df.columns])
    if args.value_col in df.columns:
        monthly = pd.to_numeric(df[args.value_col], errors="coerce")
        value_all = monthly.fillna(monthly.median()) * args.horizon * args.margin
    else:
        value_all = pd.Series(args.flat_value, index=df.index)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    yt = y_test.to_numpy()
    vt = value_all.loc[X_test.index].to_numpy()

    variants = {
        "Uncalibrated": make_pipeline(X).fit(X_train, y_train),
        "Sigmoid (Platt)": CalibratedClassifierCV(
            make_pipeline(X), method="sigmoid", cv=5).fit(X_train, y_train),
        "Isotonic": CalibratedClassifierCV(
            make_pipeline(X), method="isotonic", cv=5).fit(X_train, y_train),
    }
    probs = {k: m.predict_proba(X_test)[:, 1] for k, m in variants.items()}

    print("\n=== Calibration quality on the held-out test set ===")
    print("(lower is better for Brier, log loss, and ECE)")
    rows = []
    for name, p in probs.items():
        rows.append({"model": name,
                     "roc_auc": roc_auc_score(yt, p),
                     "brier": brier_score_loss(yt, p),
                     "log_loss": log_loss(yt, p),
                     "ece": expected_calibration_error(yt, p)})
    table = pd.DataFrame(rows).round(4)
    print(table.to_string(index=False))
    table.to_csv(os.path.join(args.outdir, "calibration_metrics.csv"), index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot([0, 1], [0, 1], ls="--", color="black", lw=1, label="Perfectly calibrated")
    for (name, p), c in zip(probs.items(), [GRAY, BLUE, GREEN]):
        frac_pos, mean_pred = calibration_curve(yt, p, n_bins=10, strategy="quantile")
        ax1.plot(mean_pred, frac_pos, marker="o", color=c, label=name)
    ax1.set_xlabel("Predicted churn probability")
    ax1.set_ylabel("Actual churn rate")
    ax1.set_title("Reliability diagram (closer to the diagonal = better)")
    ax1.legend()
    ax2.hist(probs["Uncalibrated"], bins=30, color=BLUE, alpha=0.8)
    ax2.set_xlabel("Predicted churn probability (uncalibrated model)")
    ax2.set_ylabel("Number of customers")
    ax2.set_title("How the model spreads its scores")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "13_calibration.png"), dpi=150)
    plt.close(fig)

    print("\n=== Choosing customers by expected profit (no tuned cutoff needed) ===")
    print(f"Rule: contact if  P(churn) x {args.success:.0%} save rate x value  >=  ${args.cost:,.0f}")
    n = len(yt)
    per_1000 = 1000 / n
    all_net = args.success * (yt * vt).sum() - args.cost * n
    print(f"  {'Contact everyone':<22} contacts 100% | net per 1,000: ${all_net * per_1000:>9,.0f}")
    for name, p in probs.items():
        mask = p * args.success * vt >= args.cost
        net = args.success * (yt[mask] * vt[mask]).sum() - args.cost * mask.sum()
        print(f"  {name:<22} contacts {mask.mean():>4.0%} | net per 1,000: ${net * per_1000:>9,.0f}")

    print(f"\nSaved {args.outdir}/13_calibration.png and {args.outdir}/calibration_metrics.csv")


if __name__ == "__main__":
    main()
