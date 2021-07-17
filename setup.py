from FinancialAnalysis import PROJECT_ROOT
from setuptools import setup, find_packages

import os

with open(os.path.join(PROJECT_ROOT, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="FinancialAnalysis",
    version="1.0",
    author="Yonatan Elul",
    author_email="renedal@gmail.com",
    description="A package which contains a collection of tools to download, visualize,"
                " analyze and scan stocks related data from US markets.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YonatanE8/AlgoTrading",
    package_dir={
        'FinancialAnalysis': os.getcwd(),
    },
    packages=find_packages(
        where=f'{os.getcwd()}',
        include=['FinancialAnalysis', ],
        exclude=['data']
    ),
    install_requires=[
        'beautifulsoup4',
        'numpy==1.19.3',
        'yfinance',
        'scipy',
        'statsmodels',
        'matplotlib',
        'seaborn',
        'Jupyter',
        'ipython',
        'notebook',
        'jupyterlab',
        'requests',
        'pytest',
        'html5lib',
        'plotly',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
