"""
Model comparison with cross-validation and hyperparameter tuning.

Usage:
    python churn_tuning.py --csv telco_clean.csv --target Churn --id customerID

What it does:
  1. Holds out 20% of customers as a test set that is only used once, at the end.
  2. Compares 4 models with 5-fold cross-validation on the remaining 80%:
     logistic regression, random forest, gradient boosting (default settings),
     and gradient boosting tuned with a randomized hyperparameter search.
  3. Scores every model once on the untouched test set.
  4. Saves two charts and a results table to the 'charts' folder.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BLUE, RED, GREEN, GRAY = "#4878a8", "#d95f4b", "#4a9a6b", "#8c8c8c"


def make_pipeline(X, clf):
    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


def top_decile_lift(y_true, scores):
    d = pd.DataFrame({"y": np.asarray(y_true), "s": scores}).sort_values("s", ascending=False)
    return d.head(max(1, len(d) // 10)).y.mean() / d.y.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="telco_clean.csv")
    ap.add_argument("--target", default="Churn")
    ap.add_argument("--id", default="customerID")
    ap.add_argument("--outdir", default="charts")
    ap.add_argument("--search-iter", type=int, default=30,
                    help="Number of hyperparameter combinations to try")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    y = df[args.target]
    X = df.drop(columns=[c for c in [args.target, args.id] if c in df.columns])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Train: {len(X_train):,} rows | Test (held out): {len(X_test):,} rows")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic regression": make_pipeline(X, LogisticRegression(max_iter=2000)),
        "Random forest": make_pipeline(X, RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=42)),
        "Gradient boosting (default)": make_pipeline(X, HistGradientBoostingClassifier(
            random_state=42)),
    }

    results = []
    fitted = {}

    print("\n--- 5-fold cross-validation on the training set ---")
    for name, pipe in models.items():
        cvres = cross_validate(pipe, X_train, y_train, cv=cv, n_jobs=-1,
                               scoring={"auc": "roc_auc", "ap": "average_precision"})
        results.append({"model": name,
                        "cv_auc_mean": cvres["test_auc"].mean(),
                        "cv_auc_std": cvres["test_auc"].std(),
                        "cv_pr_auc_mean": cvres["test_ap"].mean()})
        fitted[name] = pipe.fit(X_train, y_train)
        print(f"  {name:<28} AUC {cvres['test_auc'].mean():.3f} "
              f"(+/- {cvres['test_auc'].std():.3f})")

    print(f"\n--- Tuning gradient boosting ({args.search_iter} random combinations x 5 folds) ---")
    param_dist = {
        "clf__learning_rate": loguniform(0.01, 0.2),
        "clf__max_depth": randint(2, 7),
        "clf__max_iter": randint(100, 500),
        "clf__min_samples_leaf": randint(10, 80),
        "clf__l2_regularization": loguniform(1e-3, 10),
        "clf__max_leaf_nodes": [8, 15, 31],
    }
    search = RandomizedSearchCV(
        make_pipeline(X, HistGradientBoostingClassifier(random_state=42)),
        param_distributions=param_dist, n_iter=args.search_iter, cv=cv,
        scoring="roc_auc", n_jobs=-1, random_state=42, refit=True)
    search.fit(X_train, y_train)
    best_idx = search.best_index_
    name = "Gradient boosting (tuned)"
    results.append({"model": name,
                    "cv_auc_mean": search.cv_results_["mean_test_score"][best_idx],
                    "cv_auc_std": search.cv_results_["std_test_score"][best_idx],
                    "cv_pr_auc_mean": np.nan})
    fitted[name] = search.best_estimator_
    print("  Best settings found:")
    for k, v in search.best_params_.items():
        print(f"    {k.replace('clf__', '')}: {v if isinstance(v, (int, str)) else round(v, 4)}")
    print(f"  Best CV AUC: {search.best_score_:.3f}")

    # Final, one-time evaluation on the held-out test set
    print("\n--- Final scores on the untouched test set ---")
    roc_data = {}
    for r in results:
        m = fitted[r["model"]]
        p = m.predict_proba(X_test)[:, 1]
        r["test_auc"] = roc_auc_score(y_test, p)
        r["test_pr_auc"] = average_precision_score(y_test, p)
        r["test_top_decile_lift"] = top_decile_lift(y_test, p)
        roc_data[r["model"]] = roc_curve(y_test, p)
        print(f"  {r['model']:<28} AUC {r['test_auc']:.3f} | PR-AUC {r['test_pr_auc']:.3f} "
              f"| lift {r['test_top_decile_lift']:.2f}x")

    table = pd.DataFrame(results).round(4)
    table.to_csv(os.path.join(args.outdir, "model_comparison.csv"), index=False)

    # Chart 11: cross-validated AUC with error bars
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [GRAY, GRAY, BLUE, GREEN]
    labels = [m.replace(" (", "\n(") if "(" in m else m.replace(" ", "\n", 1)
              for m in table["model"]]
    ax.bar(labels, table["cv_auc_mean"], yerr=table["cv_auc_std"],
           color=colors, capsize=5)
    for i, (v, sd) in enumerate(zip(table["cv_auc_mean"], table["cv_auc_std"])):
        ax.text(i, v + sd + 0.003, f"{v:.3f}", ha="center")
    lo = max(0.5, table["cv_auc_mean"].min() - 0.06)
    ax.set_ylim(lo, table["cv_auc_mean"].max() + 0.05)
    ax.set_ylabel("ROC-AUC (5-fold cross-validation)")
    ax.set_title("Model comparison (error bars = std across folds)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "11_model_comparison.png"), dpi=150)
    plt.close(fig)

    # Chart 12: ROC curves on the test set
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for (name, (fpr, tpr, _)), c in zip(roc_data.items(), [GRAY, RED, BLUE, GREEN]):
        auc = table.loc[table.model == name, "test_auc"].iloc[0]
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="black", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the held-out test set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "12_roc_curves.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved {args.outdir}/model_comparison.csv, 11_model_comparison.png, 12_roc_curves.png")


if __name__ == "__main__":
    main()
