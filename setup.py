from src import PROJECT_ROOT
from setuptools import setup, find_packages

import os


with open(os.path.join(PROJECT_ROOT, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="AlgoTrading",
    version="1.0",
    author="Yonatan Elul",
    author_email="renedal@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YonatanE8/AlgoTrading",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
