from setuptools import setup

version = 0.1

DISTNAME = 'discrating'
LICENSE = 'BSD (3-clause)'
VERSION = version

setup(name='discrating',
      license=LICENSE,
      packages=['discrating'],
      install_requires=['numpy>=1.12', 'numba',
                        'seaborn>=0.7',
                        'joblib', 'scipy>=0.18.0', 'matplotlib>=2.0.0',
                        'scikit-learn>=1.0', 'pandas', 'ipython', 'tqdm']
      )
