from setuptools import setup, find_packages

setup(
    name="gdeltngrams",
    version="1.0.0",
    description='Python package for accessing the GDELT Web News NGrams 3.0 API and reconstructing full-text news articles.',
    author="Lorenzo Panebianco",
    license="GPL-3.0-or-later",  # SPDX identifier
    packages=find_packages(),
    install_requires=[
        "pandas",
        "requests",
        "tqdm"
    ],
    python_requires='>=3.9',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
