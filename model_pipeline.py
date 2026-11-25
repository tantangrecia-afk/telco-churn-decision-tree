"""Phase 5: reusable preprocessing and an initially untrained decision tree."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

NUMERIC_FEATURES = ["tenure", "MonthlyCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "TechSupport", "PaymentMethod"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def select_inputs(customers):
    """Use only approved columns, never the target or customer identifier."""
    missing = set(FEATURES) - set(customers.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    inputs = customers.loc[:, FEATURES].copy()
    for column in NUMERIC_FEATURES:
        inputs[column] = pd.to_numeric(inputs[column], errors="raise")
        observed = inputs[column].dropna()
        if not observed.ge(0).all() or observed.isin([float("inf"), float("-inf")]).any():
            raise ValueError(f"{column} must be finite and nonnegative.")
    if not inputs["tenure"].dropna().mod(1).eq(0).all():
        raise ValueError("tenure must be whole months.")
    # Normalize string missing values for scikit-learn's categorical imputer.
    for column in CATEGORICAL_FEATURES:
        inputs[column] = inputs[column].astype(object)
        inputs[column] = inputs[column].where(inputs[column].notna(), float("nan"))
    return inputs


def build_pipeline(**tree_settings):
    """Return a fresh pipeline; fitting later learns only from supplied rows.

    In Phase 6, cross-validation must fit the COMPLETE pipeline in each fold.
    Do not pass a matrix preprocessed using the entire training set to CV.
    """
    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop")
    settings = {"random_state": 42, "max_depth": 4, "min_samples_leaf": 20}
    settings.update(tree_settings)
    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", DecisionTreeClassifier(**settings)),
    ])
