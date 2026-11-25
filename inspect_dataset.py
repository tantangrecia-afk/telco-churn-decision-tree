"""Phase 2: inspect the original data without changing it or training a model."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "raw" / "Telco-Customer-Churn.csv"


def main():
    customers = pd.read_csv(DATA_PATH)
    print("PHASE 2: Understand the dataset")
    print(f"Source file: {DATA_PATH}")
    print(f"Customer records: {len(customers):,}")
    print(f"Columns: {len(customers.columns)}")
    print("\nCOLUMN NAMES AND LOADED TYPES")
    print(customers.dtypes.to_string())
    print("\nFIRST THREE CUSTOMERS (SELECTED COLUMNS)")
    print(customers[["customerID", "tenure", "Contract", "MonthlyCharges", "Churn"]].head(3).to_string(index=False))
    print("\nTARGET LABELS:", sorted(customers["Churn"].unique()))
    print("\nEach row represents one customer.")
    print("Churn is the target; customerID is an identifier, not a predictor.")
    print("No cleaning, splitting, or model training was performed.")


if __name__ == "__main__":
    main()
