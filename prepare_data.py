"""Phase 3: audit raw data and reserve a reproducible, stratified test set.

Blank TotalCharges become missing values, not zero. Imputation and categorical
encoding will be fitted on training data only in the Phase 5 pipeline.
"""

from pathlib import Path
import hashlib
import json

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/raw/Telco-Customer-Churn.csv"
SEED = 42


def main():
    original_hash = hashlib.sha256(RAW.read_bytes()).hexdigest()
    customers = pd.read_csv(RAW)
    duplicates = int(customers.duplicated().sum())
    duplicate_ids = int(customers["customerID"].duplicated().sum())
    if duplicates or duplicate_ids:
        raise ValueError("Duplicate records or IDs found. Review before splitting.")
    if customers["customerID"].isna().any():
        raise ValueError("Missing customer ID found.")
    if customers["Churn"].isna().any() or set(customers["Churn"].unique()) != {"Yes", "No"}:
        raise ValueError("Churn labels must be complete and contain Yes and No.")

    # This deterministic type conversion learns nothing from the data.
    total_text = customers["TotalCharges"].astype("string").str.strip()
    blank_total = total_text.isna() | total_text.eq("")
    total_numeric = pd.to_numeric(total_text, errors="coerce")
    invalid_total = total_numeric.isna() & ~blank_total
    if invalid_total.any():
        raise ValueError("Unexpected nonnumeric TotalCharges found; review raw data.")
    customers["TotalCharges"] = total_numeric

    for column in ("tenure", "MonthlyCharges", "TotalCharges"):
        observed = customers[column].dropna()
        if not observed.ge(0).all() or observed.isin([float("inf"), float("-inf")]).any():
            raise ValueError(f"Invalid numeric values in {column}.")
    if not customers["SeniorCitizen"].isin([0, 1]).all():
        raise ValueError("SeniorCitizen must be 0 or 1.")
    if not customers["tenure"].dropna().mod(1).eq(0).all():
        raise ValueError("Tenure must be whole months.")

    binary_columns = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    allowed = {column: {"Yes", "No"} for column in binary_columns}
    allowed.update({
        "gender": {"Female", "Male"},
        "MultipleLines": {"Yes", "No", "No phone service"},
        "InternetService": {"DSL", "Fiber optic", "No"},
        "Contract": {"Month-to-month", "One year", "Two year"},
        "PaymentMethod": {"Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"},
    })
    for column in ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"):
        allowed[column] = {"Yes", "No", "No internet service"}
    for column, options in allowed.items():
        if not customers[column].dropna().isin(options).all():
            raise ValueError(f"Unexpected category in {column}.")

    train, test = train_test_split(
        customers, test_size=0.20, random_state=SEED, stratify=customers["Churn"]
    )
    if set(train["customerID"]) & set(test["customerID"]):
        raise ValueError("Customer overlap between training and test sets.")
    if len(train) + len(test) != len(customers):
        raise ValueError("Split lost records.")

    destination = ROOT / "data/splits"
    destination.mkdir(parents=True, exist_ok=True)
    # IDs are retained for audit only; exclude customerID and Churn from X later.
    train.to_csv(destination / "train.csv", index=False)
    test.to_csv(destination / "test.csv", index=False)
    report = {
        "phase": 3,
        "source_sha256": original_hash,
        "rows": len(customers),
        "duplicate_rows": duplicates,
        "duplicate_customer_ids": duplicate_ids,
        "blank_total_charges": int(blank_total.sum()),
        "blank_total_charges_tenure_values": sorted(customers.loc[blank_total, "tenure"].unique().tolist()),
        "missing_after_numeric_conversion": {k: int(v) for k, v in customers.isna().sum().items() if v},
        "full_dataset_churn_counts": {k: int(v) for k, v in customers["Churn"].value_counts().items()},
        "train_rows": len(train),
        "test_rows": len(test),
        "train_churn_rate": float(train["Churn"].eq("Yes").mean()),
        "test_churn_rate": float(test["Churn"].eq("Yes").mean()),
        "random_seed": SEED,
        "test_fraction": 0.20,
        "imputation_performed": False,
        "model_trained": False,
    }
    if hashlib.sha256(RAW.read_bytes()).hexdigest() != original_hash:
        raise ValueError("Original CSV changed unexpectedly.")
    (destination / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PHASE 3: Data quality and test reservation")
    print(json.dumps(report, indent=2))
    print("PASS: No customer overlap; all records retained; original CSV unchanged.")
    print("Missing TotalCharges remain missing for Phase 5 training-only imputation.")
    print("Test records are now reserved for Phase 7. Use only train.csv in Phase 4.")


if __name__ == "__main__":
    main()
