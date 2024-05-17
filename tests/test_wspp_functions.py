# tests/test_wsppchem.py

import pytest
from wsppchem.wspp_functions import canonical_SMILES, RDkit_descriptors, load_model_and_scalers, predict_LogS, predict_logS_smiles, predict_logS_csv
import pandas as pd
import os

def test_canonical_SMILES():
    smiles = ["CCO", "O=C=O"]
    expected = ["CCO", "O=C=O"]
    result = canonical_SMILES(smiles)
    assert result == expected, f"Expected {expected}, but got {result}"

def test_RDkit_descriptors():
    smiles = ["CCO", "O=C=O"]
    descriptors, desc_names = RDkit_descriptors(smiles)
    assert len(descriptors) == len(smiles), f"Expected {len(smiles)} descriptors, but got {len(descriptors)}"
    assert len(desc_names) > 0, "Descriptor names should not be empty"

def test_load_model_and_scalers():
    model, scaler = load_model_and_scalers()
    assert model is not None, "Model should not be None"
    assert scaler is not None, "Scaler should not be None"

def test_predict_LogS():
    smiles = "CCO"
    logS = predict_LogS(smiles)
    assert isinstance(logS, float), "Predicted logS should be a float"

def test_predict_logS_smiles():
    smiles_codes = ["CCO", "O=C=O"]
    logS_values = predict_logS_smiles(*smiles_codes)
    assert len(logS_values) == len(smiles_codes), f"Expected {len(smiles_codes)} logS values, but got {len(logS_values)}"

def test_predict_logS_csv(tmp_path):
    # Create a temporary CSV file for testing
    df = pd.DataFrame({'SMILE': ['CCO', 'O=C=O']})
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

