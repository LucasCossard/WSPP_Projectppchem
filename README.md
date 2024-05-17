![Project Logo](assets/WSPP_logo.png)

<h1 align="center">
Water Solubility Prediction Project
</h1>

## Overview
This project aims to predict the water solubility of chemical compounds using machine learning techniques. The project developed here can be used to estimate the solubility of new compounds only using the SMILES code of the compounds, which can be valuable in various industries such as pharmaceuticals, agriculture, and environmental science.
In this repository, we are making available the data we used to train and test our models and `.pkl` files containing the optimized parameters of our best model. But more importantly a notebook tracing what we did from the beginning to the end of this project and a package that can predict the water solubility of several SMILEs and of a `.csv` file containing several SMILEs. 

## Project Structure
This project contains two main elements: a Notebook and a Package.

**Firstly, a Notebook containing:**
- Import Relevant Modules and Libraries
- Data Collection
- Data Cleaning
- Calculation of RDkit Molecular Descriptors
- Selection of Machine Learning Models
- Fine-tuning
- Analysis of different models
- Saving of the best trained model and scaler

**Secondly, a Package of two main functions containing:**

-  A function to predict the LogS value for one or more  SMILES
-  A function to predicts LogS values for SMILES codes stored in a CSV file
 
## Installation of the package

First, install our package `wsppchem` with a simple pip install.
```
pip install wsppchem
```

Then, you can import all the functions using the following command.
```
from wsppchem.wspp_functions import *
```

The two main functions of our package are `predict_logS_smiles` and `predict_logS_csv` which can be used in the following way:
```
predict_logS_smiles(*smiles_codes)
```
```
predict_logS_csv(csv_file_path)
```

The first function `predict_logS_smiles(*smiles_codes)` can be used to predict the LogS value for one or more SMILES at the same time.
The second fucntion `predict_logS_csv(csv_file_path)` can be used to predicts LogS values for SMILES codes stored in a CSV file.
And if you need any help, you can use the function `wspphelp()` which will give you more precise information on the functions as well as an example of how to use them. 

## License
This project is licensed under the MIT License.

## References
This project is based on the code of this Github Jupyter notebook: https://github.com/gashawmg, as well as data from https://github.com/PatWalters. 

## Authors
- Cossard Lucas: https://github.com/Nohalyan
- Venancio Enzo: https://github.com/Enzo-vnc

This project was carried out as part of EPFL's Practical programming in Chemistry course.
