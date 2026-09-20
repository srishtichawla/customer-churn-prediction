"""
Business impact of the churn model: who should we contact, and what is it worth?

Usage:
    python churn_business_impact.py --csv telco_clean.csv --target Churn --id customerID

Change the assumptions to match a scenario (all are inputs, not facts):
    --cost 50        cost of one retention offer, e.g. discount + outreach ($)
    --success 0.3    share of contacted churners the offer actually saves
    --horizon 12     months of revenue saved when a churner is retained
    --margin 0.5     profit margin on that retained revenue
    --value-col MonthlyCharges   column holding each customer's monthly revenue
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RED, BLUE, GREEN, GRAY = "#d95f4b", "#4878a8", "#4a9a6b", "#8c8c8c"


def net_curve(y, value, scores, cost, success):
    """Net benefit if we contact the top-k highest-risk customers, for k = 0..n."""
    order = np.argsort(-scores)
    gain = success * np.cumsum(y[order] * value[order])
    spend = cost * np.arange(1, len(y) + 1)
    return np.concatenate([[0.0], gain - spend]), scores[order]


def net_at_threshold(y, value, scores, thr, cost, success):
    mask = scores >= thr
    contacted = int(mask.sum())
    reached = int(y[mask].sum())
    net = success * (y[mask] * value[mask]).sum() - cost * contacted
    return contacted, reached, net


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
    ap.add_argument("--flat-value", type=float, default=400.0,
                    help="Used only if the value column is missing")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    y_all = df[args.target]
    X = df.drop(columns=[c for c in [args.target, args.id] if c in df.columns])
    if args.value_col in df.columns:
        monthly = pd.to_numeric(df[args.value_col], errors="coerce")
        value_all = monthly.fillna(monthly.median()) * args.horizon * args.margin
    else:
        value_all = pd.Series(args.flat_value, index=df.index)

    # 60% train / 20% validation (choose who to contact) / 20% test (report results)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y_all, test_size=0.2, stratify=y_all, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=42)

    num = X.select_dtypes(include="number").columns.tolist()
    cat = X.select_dtypes(exclude="number").columns.tolist()
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    model = Pipeline([("pre", pre),
                      ("clf", HistGradientBoostingClassifier(
                          max_depth=4, learning_rate=0.05, max_iter=200,
                          random_state=42))])
    model.fit(X_train, y_train)

    s_val = model.predict_proba(X_val)[:, 1]
    s_test = model.predict_proba(X_test)[:, 1]
    yv, yt = y_val.to_numpy(), y_test.to_numpy()
    vv, vt = value_all.loc[X_val.index].to_numpy(), value_all.loc[X_test.index].to_numpy()

    # Choose the risk cutoff on the validation set (never on the test set)
    net_v, sorted_v = net_curve(yv, vv, s_val, args.cost, args.success)
    k_opt = int(np.argmax(net_v))
    thr = sorted_v[k_opt - 1] if k_opt > 0 else np.inf

    n = len(yt)
    per_1000 = 1000 / n
    total_churn_value = (yt * vt).sum()

    contacted, reached, net_model = net_at_threshold(
        yt, vt, s_test, thr, args.cost, args.success)
    net_all = args.success * total_churn_value - args.cost * n
    frac = contacted / n
    net_random = frac * args.success * total_churn_value - args.cost * contacted

    print("\n=== Assumptions (edit with command-line options) ===")
    print(f"  Cost per customer contacted:     ${args.cost:,.0f}")
    print(f"  Offer success rate:              {args.success:.0%}")
    print(f"  Value of a saved churner:        {args.horizon:.0f} months x revenue x "
          f"{args.margin:.0%} margin")
    print(f"  Average value of a saved churner: ${vt[yt == 1].mean():,.0f}")

    print("\n=== Who to contact ===")
    print(f"  Risk cutoff chosen on validation data: score >= {thr:.2f}")
    print(f"  On the test set this contacts {contacted:,} of {n:,} customers ({frac:.0%})")
    print(f"  ...and reaches {reached:,} of {int(yt.sum()):,} actual churners "
          f"({reached / yt.sum():.0%} of them)")
    print(f"  Precision of the contact list: {reached / max(contacted, 1):.0%} "
          f"(overall churn rate is {yt.mean():.0%})")

    print("\n=== Net benefit per 1,000 customers (test set) ===")
    rows = [("Do nothing", 0.0),
            ("Contact everyone", net_all),
            (f"Contact {frac:.0%} at random", net_random),
            (f"Contact top {frac:.0%} by model", net_model)]
    for name, val in rows:
        print(f"  {name:<28} ${val * per_1000:>10,.0f}")

    # How sensitive is the result to the offer success rate?
    sens = []
    for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
        _, _, nm = net_at_threshold(yt, vt, s_test, thr, args.cost, s)
        na = s * total_churn_value - args.cost * n
        sens.append({"offer_success_rate": s,
                     "model_targeted_per_1000": round(nm * per_1000),
                     "contact_everyone_per_1000": round(na * per_1000)})
    sens = pd.DataFrame(sens)
    sens.to_csv(os.path.join(args.outdir, "sensitivity.csv"), index=False)
    print("\n=== Sensitivity to offer success rate (same cutoff) ===")
    print(sens.to_string(index=False))

    # Chart 9: net benefit vs. how many customers we contact
    net_t, _ = net_curve(yt, vt, s_test, args.cost, args.success)
    x = np.arange(n + 1) / n
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x * 100, net_t * per_1000, color=BLUE, lw=2, label="Contact by model risk")
    ax.plot(x * 100, x * net_all * per_1000, color=GRAY, ls="--",
            label="Contact at random")
    ax.axvline(frac * 100, color=GREEN, ls=":")
    ax.scatter([frac * 100], [net_model * per_1000], color=GREEN, zorder=5,
               label=f"Chosen cutoff ({frac:.0%} contacted)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("% of customers contacted (highest risk first)")
    ax.set_ylabel("Net benefit per 1,000 customers ($)")
    ax.set_title("How many customers should we contact?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "09_profit_curve.png"), dpi=150)
    plt.close(fig)

    # Chart 10: strategy comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [r[0] for r in rows]
    vals = [r[1] * per_1000 for r in rows]
    bars = ax.bar(names, vals, color=[GRAY, RED, GRAY, GREEN])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"${v:,.0f}",
                ha="center", va="bottom" if v >= 0 else "top")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Net benefit per 1,000 customers ($)")
    ax.set_title("Retention strategy comparison (test set)")
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "10_strategy_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"\nSaved charts/09_profit_curve.png, charts/10_strategy_comparison.png, "
          f"charts/sensitivity.csv")

    print("\n=== Resume-ready summary ===")
    def money(v):
        return f"${v:,.0f}" if v >= 0 else f"a loss of ${-v:,.0f}"

    print(f"  Under assumptions of a ${args.cost:,.0f} offer with a {args.success:.0%} save rate, "
          f"contacting only the top {frac:.0%} of customers by predicted churn risk "
          f"yields about {money(net_model * per_1000)} net benefit per 1,000 customers, "
          f"versus {money(net_all * per_1000)} for contacting everyone.")

if __name__ == "__main__":
    main()
