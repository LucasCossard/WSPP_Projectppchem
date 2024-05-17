from setuptools import setup, find_packages

setup(
    name="WSPP",
    version="0.1",
    author="Cossard Lucas and Venancio Enzo"
    packages=find_packages(),
    install_requires=[
        "python=3.8",
        "pandas",
        "numpy",
        "rdkit",
        "tqdm",
        "lightgbm",
        "requests"
    ]
)
