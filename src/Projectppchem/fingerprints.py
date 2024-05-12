import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors 
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm import tqdm
from lightgbm import LGBMRegressor
import pickle

def process_csv(file_path):
    data = pd.read_csv(file_path)
    smiles = data.iloc[:, 0].tolist()  # Assuming SMILES in the first column
    return smiles

def canonical_SMILES(smiles):
    canon_smls = [Chem.CanonSmiles(smls) for smls in smiles]
    return canon_smls

def RDkit_descriptors(smiles):
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    desc_names = calc.GetDescriptorNames()

    Mol_descriptors = []
    for mol in tqdm(mols):
        mol = Chem.AddHs(mol)
        descriptors = calc.CalcDescriptors(mol)
        Mol_descriptors.append(descriptors)
    return Mol_descriptors, desc_names

def load_model_and_scalers(model_path='model_LGBM.pkl', scaler_path='model_scaler.pkl'):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    return model, scaler

def predict_LogS(smiles):
    canonical_smiles = canonical_SMILES([smiles])
    descriptors, _ = RDkit_descriptors(canonical_smiles)

    model, scaler = load_model_and_scalers()
    scaled_descriptors = scaler.transform(descriptors)

    logS_prediction = model.predict(scaled_descriptors)
    return logS_prediction[0]

def get_logS_str():
    smiles_code = input("Enter a SMILES code: ")
    logS = predict_LogS(smiles_code)
    print(f"Predicted LogS value for {smiles_code}: {logS}")

# Example usage:
get_logS_str()
