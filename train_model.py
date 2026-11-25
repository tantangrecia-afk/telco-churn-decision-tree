"""Phase 6: training-only cross-validation, then fit the selected pipeline."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from model_pipeline import FEATURES, build_pipeline, select_inputs

ROOT = Path(__file__).resolve().parent


def main():
    training = pd.read_csv(ROOT / "data/splits/train.csv")
    inputs = select_inputs(training)
    target = training["Churn"].map({"No": 0, "Yes": 1})
    if target.isna().any():
        raise ValueError("Missing or invalid training labels.")
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(inputs, target))
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }
    # Small, declared search; selection metric is churn F1, with simpler trees
    # preferred for exact ties. The test set is never read here.
    candidates = [(depth, leaf) for depth in (3, 4, 5) for leaf in (20, 50)]
    estimators = [("baseline_always_stayed", DummyClassifier(strategy="most_frequent"), None)]
    estimators += [(f"tree_depth_{depth}_leaf_{leaf}", build_pipeline(max_depth=depth, min_samples_leaf=leaf), {"max_depth": depth, "min_samples_leaf": leaf}) for depth, leaf in candidates]
    results = []
    print("PHASE 6: Five-fold validation using training data only")
    for name, estimator, settings in estimators:
        scores = cross_validate(estimator, inputs, target, cv=folds, scoring=scoring, return_train_score=True, error_score="raise", n_jobs=1)
        row = {"candidate": name, "settings": settings}
        for metric in scoring:
            row[f"validation_{metric}_mean"] = float(scores[f"test_{metric}"].mean())
            row[f"validation_{metric}_std"] = float(scores[f"test_{metric}"].std())
        row["training_f1_mean"] = float(scores["train_f1"].mean())
        results.append(row)
        print(f"{name}: accuracy={row['validation_accuracy_mean']:.3f}, precision={row['validation_precision_mean']:.3f}, recall={row['validation_recall_mean']:.3f}, F1={row['validation_f1_mean']:.3f}")
    tree_results = results[1:]
    best = max(tree_results, key=lambda row: (row["validation_f1_mean"], -row["settings"]["max_depth"], row["settings"]["min_samples_leaf"]))
    selected = build_pipeline(**best["settings"])
    selected.fit(inputs, target)
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    model_path = models / "churn_pipeline.joblib"
    joblib.dump(selected, model_path)
    # Training-record predictions here verify serialization only, not accuracy.
    loaded = joblib.load(model_path)
    np.testing.assert_array_equal(selected.predict(inputs.head(10)), loaded.predict(inputs.head(10)))
    report = {
        "phase": 6, "training_rows": len(training), "inputs": FEATURES,
        "target_mapping": {"No": 0, "Yes": 1},
        "validation": "5-fold stratified CV; shuffle=True; seed=42; entire pipeline fitted separately within each fold",
        "selection_metric": "Mean validation F1 for churn (positive class=1)",
        "results": results, "selected": best,
        "beats_baseline_f1": best["validation_f1_mean"] > results[0]["validation_f1_mean"],
        "test_data_read": False,
        "model_path": str(model_path),
        "reload_check": "PASS: identical predictions on 10 training records; not an accuracy evaluation",
        "limitations": "Validation scores informed model selection and may be optimistic. Final generalization evaluation is reserved for Phase 7. Six-feature subset was chosen for simplicity, not established as optimal.",
    }
    output = ROOT.parent / "outputs/phase6"
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in row.items() if k != "settings"} for row in results]).to_csv(output / "validation_scores.csv", index=False)
    text = ["# Phase 6: Training and validation", "", "Five-fold validation on training data only. Churn is the positive class.", "", "| Candidate | Accuracy | Precision | Recall | F1 |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in results:
        text.append(f"| {row['candidate']} | {row['validation_accuracy_mean']:.1%} | {row['validation_precision_mean']:.1%} | {row['validation_recall_mean']:.1%} | {row['validation_f1_mean']:.3f} |")
    text += ["", f"Selected by mean churn F1: {best['candidate']}.", "", f"Mean training F1: {best['training_f1_mean']:.3f}; mean validation F1: {best['validation_f1_mean']:.3f}.", "", "The baseline predicts every customer stays. It can have high accuracy while missing every churner. Precision measures how often churn flags are correct; recall measures how many actual churners are found; F1 balances both.", "", "The complete selected pipeline was fitted on all training records and saved. Serialization verification passed. No reserved test records were read. These are validation results used for selection, not final test accuracy.", "", "Next: Phase 7, evaluate the frozen model against the reserved test set."]
    (output / "summary.md").write_text("\n".join(text), encoding="utf-8")
    print(f"\nSelected: {best['candidate']}")
    print(f"Training F1: {best['training_f1_mean']:.3f}; validation F1: {best['validation_f1_mean']:.3f}")
    print(f"Saved: {model_path}")
    print("PASS: model trained, saved, and reload verified. Reserved test data untouched.")


if __name__ == "__main__":
    main()
