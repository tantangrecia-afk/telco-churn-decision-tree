"""Phase 7: evaluate the frozen saved model; never fit or tune on test data."""

from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / "work/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from model_pipeline import select_inputs


def scores(actual, predicted):
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
    }


def main():
    model_path = ROOT / "models/churn_pipeline.joblib"
    before = hashlib.sha256(model_path.read_bytes()).hexdigest()
    model = joblib.load(model_path)
    test = pd.read_csv(ROOT / "data/splits/test.csv")
    actual = test["Churn"].map({"No": 0, "Yes": 1})
    if actual.isna().any():
        raise ValueError("Invalid test labels.")
    predicted = model.predict(select_inputs(test))
    baseline = np.zeros(len(test), dtype=int)  # Always stayed, chosen in Phase 6.
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    assert matrix.sum() == len(test)
    assert tn + fp == int(actual.eq(0).sum())
    assert fn + tp == int(actual.eq(1).sum())
    metrics = scores(actual, predicted)
    baseline_metrics = scores(actual, baseline)
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == before
    output = ROOT.parent / "outputs/phase7"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": 7, "test_rows": len(test), "model_sha256": before,
        "model_metrics": metrics, "baseline_metrics": baseline_metrics,
        "confusion_matrix_rows_actual_columns_predicted": matrix.tolist(),
        "label_order": ["Stayed", "Churned"],
        "correct_stayed": tn, "false_alarms": fp, "missed_churners": fn, "correct_churners": tp,
        "model_file_unchanged": True, "refitting_or_tuning_performed": False,
        "limitation": "Results apply to a held-out sample of the IBM dataset. They do not establish performance for a current real telecom company or Philippine customers. Further tuning on these test scores would invalidate its role as an untouched test set.",
    }
    (output / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions = test[["customerID", "Churn"]].copy()
    predictions.columns = ["customerID", "actual_churn"]
    predictions["predicted_churn"] = np.where(predicted == 1, "Yes", "No")
    predictions.to_csv(output / "test_predictions.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ConfusionMatrixDisplay(matrix, display_labels=["Stayed", "Churned"]).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Reserved test set: {len(test):,} customers")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    summary = ["# Phase 7: Final test evaluation", "", "The saved decision tree was evaluated without refitting or tuning.", "", "| Metric | Always stayed baseline | Decision tree |", "| --- | ---: | ---: |"]
    for metric in metrics:
        summary.append(f"| {metric.title()} | {baseline_metrics[metric]:.1%} | {metrics[metric]:.1%} |")
    summary += ["", f"- Correctly predicted stayed: {tn}.", f"- Correctly identified churners: {tp}.", f"- False alarms (stayed, predicted churn): {fp}.", f"- Missed churners (churned, predicted stayed): {fn}.", "", "Precision: fraction of churn flags that were correct. Recall: fraction of actual churners detected. F1 balances precision and recall.", "", report["limitation"], "", "Checks passed: confusion-matrix totals match test labels; model file unchanged. No model fitting or tuning performed.", "", "Next: Phase 8, create a reusable prediction script and demonstrate unseen customer predictions."]
    (output / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("PHASE 7: Reserved test evaluation")
    print(json.dumps(report, indent=2))
    print("PASS: Evaluation complete; saved model unchanged.")


if __name__ == "__main__":
    main()
