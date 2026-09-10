
from setuptools import setup, find_packages

setup(
    name="ur_env",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
    ],
    python_requires=">=3.8",
)
