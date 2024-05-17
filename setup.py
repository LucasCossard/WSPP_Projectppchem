from setuptools import setup, find_packages

setup(
    name="WSPP_Projectppchem",
    version="0.1",
    author="Cossard Lucas and Venancio Enzo",
    packages=find_packages(),
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
