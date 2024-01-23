from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="scp", 
    version="0.1",
    description="A generative model for single-cell proteomics",
    packages=find_packages(),
    install_requires=requirements,
)


"""
install_requires=[
    
    "scvi-tools",
    "lightning==2.0.1", # issue with scvi-tools and newer versions of lightning
    "scanpy",
    #"anndata2ri",
    #"rpy2==3.4.2",
]
"""