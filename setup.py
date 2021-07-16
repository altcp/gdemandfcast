#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', 'arch>=4.15', 'pmdarima>=1.8.2', 'scikit-learn>=0.24.2', 'keras-tuner>=1.0.1', 'tensorflow>=2.4.1',
'pandas>=1.2.5', 'xgboost>=1.4.0', 'matplotlib>=3.3.3']

test_requirements = ['pytest>=3.8', ]

setup(
    author="Tan Chiang Pern Alvin",
    author_email='colab.tcp@gmail.com',
    python_requires='>=3.8.5',
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8.5',
    ],
    description="A Generic Python Package for Demand Forecasting.",
    entry_points={
        'console_scripts': [
            'gdemandfcast=gdemandfcast.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='gdemandfcast',
    name='gdemandfcast',
    packages=find_packages(include=['gdemandfcast', 'gdemandfcast.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/altcp/gdemandfcast',
    version='0.1.0',
    zip_safe=False,
)
