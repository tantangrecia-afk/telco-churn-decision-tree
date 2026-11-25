"""Phase 8: interactive predictions or a reproducible held-out demonstration."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from model_pipeline import FEATURES, select_inputs

ROOT = Path(__file__).resolve().parent
OPTIONS = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
}


def validate_customer(customer):
    """Require complete, valid user entries; never silently guess an input."""
    missing = set(FEATURES) - set(customer)
    if missing:
        raise ValueError(f"Missing required inputs: {sorted(missing)}")
    for column in FEATURES:
        if pd.isna(customer[column]) or str(customer[column]).strip() == "":
            raise ValueError(f"{column} is required.")
    for column, choices in OPTIONS.items():
        if customer[column] not in choices:
            raise ValueError(f"Invalid {column}. Choose one of: {choices}")
    if customer["InternetService"] == "No" and customer["TechSupport"] != "No internet service":
        raise ValueError("Without internet service, TechSupport must be 'No internet service'.")
    if customer["InternetService"] != "No" and customer["TechSupport"] == "No internet service":
        raise ValueError("With internet service, TechSupport must be Yes or No.")
    return select_inputs(pd.DataFrame([customer]))


def predict_one(model, customer):
    inputs = validate_customer(customer)
    return "Yes" if int(model.predict(inputs)[0]) == 1 else "No"


def ask_number(label, integer=False):
    while True:
        text = input(f"{label}: ").strip()
        try:
            number = float(text)
            if not 0 <= number < float("inf") or (integer and not number.is_integer()):
                raise ValueError
            return int(number) if integer else number
        except ValueError:
            print("Enter a finite, nonnegative " + ("whole number." if integer else "number."))


def ask_choice(column):
    choices = OPTIONS[column]
    print(f"\n{column}:")
    for i, value in enumerate(choices, 1):
        print(f"  {i}. {value}")
    while True:
        choice = input("Enter the option number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(choices):
            return choices[int(choice) - 1]
        print(f"Choose a number from 1 to {len(choices)}.")


def demo(model):
    # Fixed first five test records, selected without looking at predictions.
    examples = pd.read_csv(ROOT / "data/splits/test.csv").head(5)
    results = []
    for _, row in examples.iterrows():
        customer = row[FEATURES].to_dict()
        predicted = predict_one(model, customer)
        results.append({"customerID": row["customerID"], **customer, "actual_churn": row["Churn"], "predicted_churn": predicted, "correct": predicted == row["Churn"]})
    table = pd.DataFrame(results)
    print("\nDEMO: first five test records (never used to train the model)")
    print(table[["customerID", "actual_churn", "predicted_churn", "correct"]].to_string(index=False))
    output = ROOT.parent / "outputs/phase8"
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "demo_predictions.csv", index=False)
    lines = ["# Phase 8: Saved-model prediction demo", "", "Examples: first five reserved test records, selected without inspecting predictions. Actual labels are used only for comparison. These five examples are not an accuracy estimate; see Phase 7 for full evaluation.", "", "| Customer | Actual churn | Predicted churn | Correct |", "| --- | --- | --- | --- |"]
    lines += [f"| {r['customerID']} | {r['actual_churn']} | {r['predicted_churn']} | {r['correct']} |" for r in results]
    lines += ["", "Interactive use from the project terminal:", "", "```powershell", r".\.venv\Scripts\python.exe .\predict_customer.py", "```", "", "Reproduce this demo:", "", "```powershell", r".\.venv\Scripts\python.exe .\predict_customer.py --demo", "```", "", "Enter tenure in months, monthly charges in the dataset's monetary units, and choose the four service/billing categories. No training takes place. Predictions are uncertain; the model can make both false alarms and missed-churn errors."]
    (output / "demo_guide.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Predict customer churn using the saved pipeline.")
    parser.add_argument("--demo", action="store_true", help="Show predictions for the first five test records.")
    args = parser.parse_args()
    model_path = ROOT / "models/churn_pipeline.joblib"
    if not model_path.exists():
        parser.error("Saved model missing. Complete Phase 6 first.")
    model = joblib.load(model_path)
    print("PHASE 8: Saved-model customer prediction (no retraining)")
    if args.demo:
        demo(model)
        return
    print("Enter a customer's details. Prediction is an estimate, not a certainty.")
    customer = {
        "tenure": ask_number("Tenure (whole months)", integer=True),
        "MonthlyCharges": ask_number("Monthly charges (dataset monetary units)"),
        "Contract": ask_choice("Contract"),
        "InternetService": ask_choice("InternetService"),
    }
    customer["TechSupport"] = "No internet service" if customer["InternetService"] == "No" else ask_choice("TechSupport")
    while customer["InternetService"] != "No" and customer["TechSupport"] == "No internet service":
        print("Internet customers must choose Yes or No for technical support.")
        customer["TechSupport"] = ask_choice("TechSupport")
    customer["PaymentMethod"] = ask_choice("PaymentMethod")
    prediction = predict_one(model, customer)
    print("\nCUSTOMER INPUTS")
    for column in FEATURES:
        print(f"{column}: {customer[column]}")
    print(f"\nPredicted churn: {prediction} ({'likely to leave' if prediction == 'Yes' else 'likely to stay'})")
    print("This is a model prediction, not a known customer outcome.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, EOFError, KeyboardInterrupt) as error:
        print(f"\nPrediction stopped: {error}")
        raise SystemExit(1)
