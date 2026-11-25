"""Phase 5 checks: fit a disposable preprocessor on training data only."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from model_pipeline import FEATURES, NUMERIC_FEATURES, build_pipeline, select_inputs

ROOT = Path(__file__).resolve().parent


def main():
    training = pd.read_csv(ROOT / "data/splits/train.csv")
    inputs = select_inputs(training)
    assert inputs.columns.tolist() == FEATURES
    assert "Churn" not in inputs and "customerID" not in inputs
    pipeline = build_pipeline()
    preprocessing = pipeline.named_steps["preprocessing"]
    matrix = preprocessing.fit_transform(inputs)
    assert matrix.shape[0] == len(training)
    assert np.isfinite(matrix).all()
    learned_medians = preprocessing.named_transformers_["numeric"].named_steps["impute"].statistics_
    np.testing.assert_allclose(learned_medians, inputs[NUMERIC_FEATURES].median().to_numpy())

    # Synthetic probes are checks, not real customer outcomes or accuracy tests.
    probe = inputs.iloc[[0]].copy()
    probe["tenure"] = np.nan
    probe["MonthlyCharges"] = np.nan
    probe["TechSupport"] = np.nan
    transformed = preprocessing.transform(select_inputs(probe))
    assert transformed.shape[1] == matrix.shape[1] and np.isfinite(transformed).all()
    np.testing.assert_allclose(transformed[0, :2], learned_medians)
    probe = inputs.iloc[[0]].copy()
    probe.loc[:, "PaymentMethod"] = "Synthetic new payment category"
    transformed = preprocessing.transform(select_inputs(probe))
    assert transformed.shape[1] == matrix.shape[1] and np.isfinite(transformed).all()
    np.testing.assert_allclose(preprocessing.named_transformers_["numeric"].named_steps["impute"].statistics_, learned_medians)
    try:
        select_inputs(inputs.drop(columns="Contract"))
    except ValueError:
        pass
    else:
        raise AssertionError("Missing required column was not rejected.")
    try:
        check_is_fitted(pipeline.named_steps["model"])
    except NotFittedError:
        pass
    else:
        raise AssertionError("Classifier should not be trained in Phase 5.")

    report = {
        "phase": 5,
        "status": "PASS",
        "inputs": FEATURES,
        "training_rows": len(training),
        "encoded_columns": matrix.shape[1],
        "missing_selected_inputs": inputs.isna().sum().to_dict(),
        "numeric_training_medians": dict(zip(NUMERIC_FEATURES, learned_medians.tolist())),
        "encoded_feature_names": preprocessing.get_feature_names_out().tolist(),
        "checks": ["target and ID excluded", "finite output", "training-only medians", "missing values handled", "unseen category handled", "missing column rejected", "classifier remains untrained"],
        "test_data_read": False,
        "classifier_trained": False,
        "note": "Disposable preprocessing check only. Phase 6 must fit a fresh complete pipeline inside each validation fold. Unknown categories encode as zeros; this does not guarantee reliable predictions for new categories.",
    }
    destination = ROOT.parent / "outputs/phase5"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "preprocessing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PHASE 5: Preprocessing verification")
    print(json.dumps(report, indent=2))
    print("PASS: Preprocessing works. No classifier trained; test data untouched.")


if __name__ == "__main__":
    main()
