# tests/test_wsppchem.py

import pytest
from wsppchem.wspp_functions import (
    canonical_SMILES,
    RDkit_descriptors,
    load_model_and_scalers,
    predict_LogS,
    predict_logS_smiles,
    predict_logS_csv,
)

def test_canonical_SMILES():
    smiles = ["CCO", "O=C=O"]
    expected = ["CCO", "O=C=O"]
    assert canonical_SMILES(smiles) == expected

def test_RDkit_descriptors():
    smiles = ["CCO", "O=C=O"]
    descriptors, desc_names = RDkit_descriptors(smiles)
    assert len(descriptors) == 2
    assert len(desc_names) > 0

def test_load_model_and_scalers():
    model, scaler = load_model_and_scalers()
    assert model is not None
    assert scaler is not None

def test_predict_LogS():
    smiles = "CCO"
    logS = predict_LogS(smiles)
    assert isinstance(logS, float)

def test_predict_logS_smiles():
    smiles_codes = ["CCO", "O=C=O"]
    results = predict_logS_smiles(*smiles_codes)
    assert len(results) == 2

def test_predict_logS_csv(tmp_path):
    csv_content = "SMILE\nCCO\nO=C=O\n"
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)
    predict_logS_csv(str(csv_file))
    output_file = tmp_path / "test_predicted.csv"
    assert output_file.exists()

