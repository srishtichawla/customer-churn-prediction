"""
Churn analysis: exploratory charts + model explainability (SHAP).

Usage:
    python churn_analysis.py --csv telco_clean.csv --target Churn --id customerID

All charts are saved as PNG files in the 'charts' folder.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    import shap
except ImportError:
    shap = None

RED, BLUE = "#d95f4b", "#4878a8"


def save(fig, outdir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {outdir}/{name}")


def run_eda(df, target, id_col, outdir):
    y = df[target]
    print("\n--- Exploratory analysis ---")

    # 1. Overall churn split
    fig, ax = plt.subplots(figsize=(4.5, 4))
    counts = y.value_counts().sort_index()
    ax.bar(["Stayed", "Churned"], counts.values, color=[BLUE, RED])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / len(y):.1%})", ha="center", va="bottom")
    ax.set_title("Customers who stayed vs. churned")
    ax.set_ylim(0, counts.max() * 1.18)
    save(fig, outdir, "01_churn_split.png")

    # 2. Churn rate by the categorical features with the biggest differences
    cats = [c for c in df.select_dtypes(exclude="number").columns
            if c != id_col and df[c].nunique() <= 6]
    spread = {c: df.groupby(c)[target].mean().agg(lambda s: s.max() - s.min())
              for c in cats}
    top = sorted(spread, key=spread.get, reverse=True)[:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, c in zip(axes.ravel(), top):
        rates = df.groupby(c)[target].mean().sort_values(ascending=False) * 100
        ax.bar(rates.index.astype(str), rates.values, color=RED)
        ax.axhline(y.mean() * 100, ls="--", color="gray", label="overall")
        ax.set_title(c)
        ax.set_ylabel("Churn rate (%)")
        ax.tick_params(axis="x", rotation=20)
    for ax in axes.ravel()[len(top):]:
        ax.axis("off")
    fig.suptitle("Churn rate by segment (dashed line = overall churn rate)", y=1.02)
    save(fig, outdir, "02_churn_by_segment.png")
    for c in top[:3]:
        rates = df.groupby(c)[target].mean()
        print(f"  {c}: highest churn = '{rates.idxmax()}' ({rates.max():.1%}), "
              f"lowest = '{rates.idxmin()}' ({rates.min():.1%})")

    # 3. Numeric distributions, churned vs. stayed
    nums = [c for c in df.select_dtypes(include="number").columns
            if c not in (target, id_col) and df[c].nunique() > 10]
    if nums:
        fig, axes = plt.subplots(1, len(nums), figsize=(5 * len(nums), 4), squeeze=False)
        for ax, c in zip(axes[0], nums):
            ax.hist(df.loc[y == 0, c].dropna(), bins=30, alpha=0.6, density=True,
                    color=BLUE, label="Stayed")
            ax.hist(df.loc[y == 1, c].dropna(), bins=30, alpha=0.6, density=True,
                    color=RED, label="Churned")
            ax.set_title(f"{c} distribution")
            ax.legend()
        save(fig, outdir, "03_numeric_distributions.png")

        # 4. Churn rate across deciles of the first numeric feature (e.g. tenure)
        col = "tenure" if "tenure" in nums else nums[0]
        bins = pd.qcut(df[col], 10, duplicates="drop")
        rate = df.groupby(bins, observed=True)[target].mean() * 100
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(len(rate)), rate.values, marker="o", color=RED)
        ax.set_xticks(range(len(rate)))
        ax.set_xticklabels([f"{i.left:.0f}-{i.right:.0f}" for i in rate.index],
                           rotation=30)
        ax.set_xlabel(f"{col} (grouped into deciles)")
        ax.set_ylabel("Churn rate (%)")
        ax.set_title(f"Churn rate by {col}")
        save(fig, outdir, "04_churn_by_numeric.png")


def run_model_explainability(df, target, id_col, outdir):
    print("\n--- Model + explainability ---")
    y = df[target]
    X = df.drop(columns=[c for c in [target, id_col] if c in df.columns])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]), cat),
    ], verbose_feature_names_out=False)
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=42)
    pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(X_train, y_train)
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    print(f"  Test ROC-AUC: {auc:.3f}")

    # Permutation importance on the original columns (no extra libraries needed)
    imp = permutation_importance(pipe, X_test, y_test, scoring="roc_auc",
                                 n_repeats=5, random_state=42)
    ranked = pd.Series(imp.importances_mean, index=X_test.columns).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(ranked.index, ranked.values, color=BLUE)
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title("Which features matter most?")
    save(fig, outdir, "05_feature_importance.png")

    if shap is None:
        print("  SHAP not installed, skipping SHAP charts (pip install shap).")
        return

    # SHAP: how each feature pushes an individual prediction up or down
    Xte = pipe.named_steps["pre"].transform(X_test)
    names = pipe.named_steps["pre"].get_feature_names_out()
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xte)
    if isinstance(sv, list):
        sv = sv[1]

    shap.summary_plot(sv, Xte, feature_names=names, show=False, max_display=12)
    plt.title("SHAP: red = high feature value, right = pushes toward churn")
    save(plt.gcf(), outdir, "06_shap_summary.png")

    shap.summary_plot(sv, Xte, feature_names=names, plot_type="bar",
                      show=False, max_display=12)
    plt.title("Average impact of each feature on churn risk (SHAP)")
    save(plt.gcf(), outdir, "07_shap_bar.png")

    # Explain the single highest-risk customer in the test set
    idx = int(np.argmax(pipe.predict_proba(X_test)[:, 1]))
    base = float(np.ravel(explainer.expected_value)[0])
    expl = shap.Explanation(values=sv[idx], base_values=base,
                            data=Xte[idx], feature_names=names)
    shap.plots.waterfall(expl, show=False, max_display=10)
    plt.title("Why is this customer the highest churn risk?")
    save(plt.gcf(), outdir, "08_shap_top_customer.png")

    mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False)
    print("\n  Top drivers of churn risk (mean |SHAP|):")
    for f, v in mean_abs.head(5).items():
        print(f"    {f}: {v:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="telco_clean.csv")
    ap.add_argument("--target", default="Churn")
    ap.add_argument("--id", default="customerID")
    ap.add_argument("--outdir", default="charts")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} rows | churn rate: {df[args.target].mean():.1%}")
    run_eda(df, args.target, args.id, args.outdir)
    run_model_explainability(df, args.target, args.id, args.outdir)
    print(f"\nDone. Open the '{args.outdir}' folder to see your charts.")


if __name__ == "__main__":
    main()
