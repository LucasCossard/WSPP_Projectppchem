import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors 
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm import tqdm
from lightgbm import LGBMRegressor
import pickle
import os

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

def load_model_and_scalers(model_path=None, scaler_path=None):

    model_full_path = "/content/Projectppchem/Data/LGBMRegressor/model_LGBM.pkl" #Provide here the path to the model_LGBM.pkl file
    scaler_full_path = "/content/Projectppchem/Data/LGBMRegressor/scaler_LGBM.pkl" #Provide here the path to the scaler_LGBM.pkl file

    with open(model_full_path, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(scaler_full_path, 'rb') as scaler_file:
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
    print(
        """\
  ***********************************
  *    _______          _______     *
  *    / ____\ \        / /  __ \   *
  *   | (___  \ \  /\  / /| |__) |  *
  *    \___ \  \ \/  \/ / |  ___/   *
  *    ____) |  \  /\  /  | |       *
  *   |_____/    \/  \/   |_|       *
  *                                 *
  ***********************************"""
   )
     smiles_code = input("Please enter a SMILES code :")
                                
     logS = predict_LogS(smiles_code)
     print(f"Predicted LogS value for {smiles_code}: {logS}")

# Example usage:
get_logS_str()
