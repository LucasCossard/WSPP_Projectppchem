#  Water Solubility Prediction Project

## Overview
This project aims to predict the water solubility of chemical compounds using machine learning techniques. The project developed here can be used to estimate the solubility of new compounds only using the SMILES code of the compounds, which can be valuable in various industries such as pharmaceuticals, agriculture, and environmental science.
In this repository, we are making available the data we used to train and test our models, `.pmd` files containing the optimized parameters of our models, and a notebook tracing what we did from the beginning to the end of this project and a package . 

## Project Structure
The project is structured as follows:
**First, a Notebook containing :**
- Import Relevant Modules and Libraries
- Data Collection
- Data Cleaning
- Calculation of RDkit Molecular Descriptors
- Select Machine Learning Models
- Fine-tuning

## Installation
1. Clone this repository:
```
git clone https://github.com/Nohalyan/Projetppchem
```
2. Install the required Python packages:
```
!pip install pathlib numpy pandas rdkit matplotlib scikit-learn lightgbm lazypredict tqdm
#Remplacer par un package ?
```

## Usage
1. **Data Preparation:** Place your dataset in the `data/` directory. Ensure the dataset is formatted correctly with features and labels.
2. **Exploratory Data Analysis:** Explore the dataset using the Colab notebooks in the `notebooks/` directory to understand the data distribution and relationships.
3. **Model Training:** Use the scripts in the Colab notebooks in the `notebooks/` directory to preprocess the data, train machine learning models, and save the trained models in the corresponding `models/` directory.
4. **Model Evaluation:** Evaluate the model performance using the evaluation using the scripts in the Colab notebooks in the `notebooks/` directory.
5. **Prediction:** Once trained, the models in the models/ directory can be used to predict the water solubility of new compounds by providing the required input features.
```
Changer ici par le code qui utilsie que un smile oui le csv
```
## License
This project is licensed under the MIT License.

## References
This project is based on the code of this Github Jupyter notebook: https://github.com/gashawmg, as well as data from https://github.com/PatWalters. 

## Authors
- Cossard Lucas: https://github.com/Nohalyan
- Venancio Enzo: https://github.com/Enzo-vnc

This project was carried out as part of EPFL's Practical programming in Chemistry course.
