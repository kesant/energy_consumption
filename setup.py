import os

from setuptools import setup, find_packages

def readme() -> str:
    """Utility function to read the README.md.

    Used for the `long_description`. It's nice, because now
    1) we have a top level README file and
    2) it's easier to type in the README file than to put a raw string in below.

    Args:
        nothing

    Returns:
        String of readed README.md file.
    """
    return open(os.path.join(os.path.dirname(__file__), 'README.md')).read()

setup(
    name='energy_consumption_architecture',
    version='0.1.0',
    author='kevin santana',
    author_email='kejosant@espol.edu.ec',
    description='The project involves an energy consumption prediction system for non-residential buildings, utilizing clustering techniques and machine learning models. The goal is to improve prediction accuracy by grouping buildings with similar consumption patterns. Representative time series for each cluster (average and the one closest to the centroid) are evaluated. Several machine learning models are compared, and the best ones are selected based on the root mean square error (RMSE) to predict total energy consumption.',
    python_requires='>=3',
    url='',
    packages=find_packages(),
    long_description=readme(),
)