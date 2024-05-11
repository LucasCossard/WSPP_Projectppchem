# import the necessary libraries
import numpy as np
import pandas as pd
from matplotlib import image, pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sn

from sklearn.metrics import r2_score, mean_squared_error
import pickle

from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors 
from rdkit.ML.Descriptors import MoleculeDescriptors

def process_csv(file_path):
    data = pd.read_csv(file_path)
    smiles = data.iloc[:, 0].tolist()  # Assuming SMILES in the first column
    return smiles
'''
Reads a CSV file from the specified file_path assuming that SMILES codes are in the first column of the CSV file.
Returns a list of SMILES codes extracted from the CSV file.

'''
def canonical_SMILES(smiles):
    canon_smls = [Chem.CanonSmiles(smls) for smls in smiles]
    return canon_smls
'''
Takes a list of SMILES codes as input "smiles" adn generates the canonical SMILES representation for each SMILES code using RDKit's "Chem.CanonSmiles" function.
Returns a list of canonical SMILES codes.

'''

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

'''
Takes a list of SMILES codes as input "smiles", converts each SMILES code into an RDKit molecule object and calculates RDKit descriptors for each molecules.u
Returns a tuple containing: list of descriptor values for each molecule "Mol_descriptors" and a list of descriptor names "desc_names".

'''
#Add the .pkl files paths
def load_model_and_scalers(model_path='model_LGBM.pkl', scaler_path='model_scaler.pkl'):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    return model, scaler
    
#===================================================================================================================================================================================================
