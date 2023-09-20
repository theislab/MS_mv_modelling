from setuptools import setup, find_packages

setup(
    name="scp", 
    version="0.1",
    description="A generative model for single-cell proteomics",
    packages=find_packages(),

    install_requires=[
        "scvi-tools",
        "lightning==2.0.1", # issue with scvi-tools and newer versions of lighting
    ]

)