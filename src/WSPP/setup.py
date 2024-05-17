from setuptools import setup, find_packages

setup(
    name="WSPP",
    version="0.1",
    author="Cossard Lucas and Venancio Enzo"
    packages=find_packages(),
    install_requires=[
        "requests",
    ],
)
