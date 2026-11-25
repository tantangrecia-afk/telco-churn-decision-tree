"""Phase 4: descriptive analysis of TRAINING data only. No model training."""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT.parent / "outputs" / "phase4"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / "work/matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd


def main():
    customers = pd.read_csv(ROOT / "data/splits/train.csv")
    customers["churned"] = customers["Churn"].eq("Yes")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    overall = customers["churned"].mean()
    print("PHASE 4: Explore training data only")
    print(f"Training customers: {len(customers):,}; churn rate: {overall:.1%}")
    findings = ["# Phase 4: Training-data exploration", "", f"Training records: {len(customers):,}. Overall churn rate: {overall:.1%}.", "", "These are descriptive associations, not evidence of causes. Test data was not used.", ""]

    customers["TenureGroup"] = pd.cut(customers["tenure"], [-1, 12, 24, 48, float("inf")], labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"])
    groups = [
        ("Contract", ["Month-to-month", "One year", "Two year"], "Churn by contract type", "contract"),
        ("TenureGroup", ["0-12 months", "13-24 months", "25-48 months", "49+ months"], "Churn by customer tenure", "tenure"),
        ("InternetService", ["DSL", "Fiber optic", "No"], "Churn by internet service", "internet_service"),
    ]
    for column, order, title, filename in groups:
        summary = customers.groupby(column, observed=True)["churned"].agg(customers="size", churned="sum", churn_rate="mean").reindex(order)
        assert summary["customers"].sum() == len(customers)
        assert summary["churned"].sum() == customers["churned"].sum()
        summary.to_csv(OUTPUT / f"{filename}_summary.csv")
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(summary.index.astype(str), summary["churn_rate"], color="#247c9c", width=0.6)
        ax.set(title=title, ylabel="Customers who churned", ylim=(0, max(0.5, summary["churn_rate"].max() + 0.12)))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.axhline(overall, color="#bb5b32", linestyle="--", label=f"Overall training churn: {overall:.1%}")
        for bar, (_, row) in zip(bars, summary.iterrows()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.012, f"{row['churn_rate']:.1%}\nn={int(row['customers']):,}", ha="center", va="bottom", fontsize=10)
        ax.legend(loc="upper right", frameon=False)
        fig.text(0.5, 0.015, "IBM Telco sample | Training data only | Association does not establish causation", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(OUTPUT / f"{filename}.png", dpi=150)
        plt.close(fig)
        print(f"\n{title}\n{summary.to_string()}")
        findings.append(f"## {title}")
        for label, row in summary.iterrows():
            findings.append(f"- {label}: {row['churn_rate']:.1%} churn ({int(row['churned']):,} of {int(row['customers']):,} customers).")
        findings.append("")

    charge_summary = customers.groupby("Churn")["MonthlyCharges"].agg(customers="size", mean="mean", median="median")
    charge_summary.to_csv(OUTPUT / "monthly_charges_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([customers.loc[customers.Churn.eq(label), "MonthlyCharges"] for label in ["No", "Yes"]], tick_labels=["Stayed", "Churned"], patch_artist=True, boxprops={"facecolor": "#9dcbd7"}, medianprops={"color": "#a43c20"})
    ax.set(title="Monthly charges by churn outcome", ylabel="Monthly charges (dataset units)")
    fig.text(0.5, 0.015, "Training data only | Line = median; box = middle 50%; whiskers = up to 1.5 x IQR", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUTPUT / "monthly_charges.png", dpi=150)
    plt.close(fig)
    print(f"\nMonthly charges\n{charge_summary.to_string()}")
    findings.extend(["## Monthly charges", ""])
    for label, row in charge_summary.iterrows():
        findings.append(f"- Churn={label}: median {row['median']:.2f}, mean {row['mean']:.2f} (dataset monetary units).")
    findings.extend(["", "## Interpretation limits", "", "Group differences can reflect several related factors. These comparisons do not isolate a cause of churn. This sample does not establish performance on current or local telecom customers.", "", "Next: Phase 5, agree on inputs and build training-only preprocessing. No model has been trained."])
    (OUTPUT / "findings.md").write_text("\n".join(findings), encoding="utf-8")
    print(f"\nPASS: group totals verified. Charts and findings saved to {OUTPUT}")


if __name__ == "__main__":
    main()
