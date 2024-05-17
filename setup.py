from setuptools import setup, find_packages

setup(
    name="WSPProjectppchem",
    version="0.1",
    author="Cossard Lucas and Venancio Enzo",
    packages=find_packages(),
    description = "water solubility prediction project",
    long_description = file: README.md,
    url="https://github.com/Nohalyan/WSPP_Projectppchem",
    python_requires='>=3.8', 
    install_requires=[
        "pandas",
        "numpy",
        "rdkit",
        "tqdm",
        "lightgbm",
        "requests"
    ],
)
