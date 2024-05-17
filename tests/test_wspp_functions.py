# tests/test_wsppchem.py
"""
First test summary:
===================================== short test summary info ======================================
FAILED tests/test_wspp_functions.py::test_load_model_and_scalers - ModuleNotFoundError: No module named 'sklearn'
FAILED tests/test_wspp_functions.py::test_predict_LogS - ModuleNotFoundError: No module named 'sklearn'
FAILED tests/test_wspp_functions.py::test_predict_logS_smiles - ModuleNotFoundError: No module named 'sklearn'
FAILED tests/test_wspp_functions.py::test_predict_logS_csv - ModuleNotFoundError: No module named 'sklearn'
=================================== 4 failed, 2 passed in 0.83s ====================================
py310: exit 1 (1.10 seconds) /content/WSPP_Projectppchem> pytest pid=1575
  py38: SKIP (0.09 seconds)
  py39: SKIP (0.01 seconds)
  py310: FAIL code 1 (26.98=setup[25.88]+cmd[1.10] seconds)
  evaluation failed :( (27.16 seconds)
"""

import pytest
from wsppchem.wspp_functions import canonical_SMILES, RDkit_descriptors, load_model_and_scalers, predict_LogS, predict_logS_smiles, predict_logS_csv
import pandas as pd
import os
import sklearn

def test_canonical_SMILES():
    smiles = ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(=O)NC1=CC=C(C=C1)O"]
    expected = ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(=O)NC1=CC=C(C=C1)O"]
    result = canonical_SMILES(smiles)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_RDkit_descriptors():
    smiles = ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(=O)NC1=CC=C(C=C1)O"]
    descriptors, desc_names = RDkit_descriptors(smiles)
    assert len(descriptors) == len(smiles), f"Expected {len(smiles)} descriptors, but got {len(descriptors)}"
    assert len(desc_names) > 0, "Descriptor names should not be empty"

def test_load_model_and_scalers():
    model, scaler = load_model_and_scalers()
    assert model is not None, "Model should not be None"
    assert scaler is not None, "Scaler should not be None"

def test_predict_LogS():
    smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    logS = predict_LogS(smiles)
    assert isinstance(logS, float), "Predicted logS should be a float"

def test_predict_logS_smiles():
    smiles_codes = ["CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(=O)NC1=CC=C(C=C1)O"]
    logS_values = predict_logS_smiles(*smiles_codes)
    assert len(logS_values) == len(smiles_codes), f"Expected {len(smiles_codes)} logS values, but got {len(logS_values)}"

def test_predict_logS_csv(tmp_path):
    # Create a temporary CSV file for testing
    df = pd.DataFrame({'SMILE': ['CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'CC(=O)NC1=CC=C(C=C1)O']})
    test_csv_file = tmp_path / "test_smiles.csv"
    df.to_csv(test_csv_file, index=False)

    predict_logS_csv(test_csv_file)

    predicted_csv_file = tmp_path / "test_smiles_predicted.csv"
    assert predicted_csv_file.exists(), f"Expected output CSV file {predicted_csv_file} does not exist"

    # Read the predicted CSV file and check the contents
    df_predicted = pd.read_csv(predicted_csv_file)
    assert 'Predicted_LogS mol/L' in df_predicted.columns, "Predicted CSV file should contain 'Predicted_LogS mol/L' column"
    assert len(df_predicted) == len(df), f"Expected {len(df)} rows, but got {len(df_predicted)}"

# Run the tests using pytest
if __name__ == "__main__":
    pytest.main()

