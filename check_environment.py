"""Phase 1: verify the Python environment before working with customer data."""

import sys

import joblib
import matplotlib
import pandas
import seaborn
import sklearn

print("PHASE 1: Environment check")
print(f"Python: {sys.version.split()[0]}")
print(f"Interpreter: {sys.executable}")
print(f"Virtual environment: {sys.prefix != sys.base_prefix}")
for package in (pandas, matplotlib, seaborn, sklearn, joblib):
    print(f"{package.__name__}: {package.__version__}")
print("SUCCESS: All required packages imported.")
