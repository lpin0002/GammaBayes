from setuptools import setup, find_packages
import subprocess
import sys
import os


setup(name='GammaBayes',
      description='A package for Bayesian dark matter inference',
      url='https://github.com/lpin0002/GammaBayes',
      author='Liam Pinchbeck',
      author_email='Liam.Pinchbeck@monash.edu',
      license="MIT",
      version='0.0.41',
      packages=find_packages(),
      install_requires=[
          "astropy>=5.1",
        "corner>=2.2.2",
        "dynesty==2.1.2",
        "jupyterlab>=3.6.3",
        "matplotlib==3.7.1",
        "scipy==1.10.1",
        "tqdm>=4.65.0",
        "numpy >=1.23",
        "gammapy==0.20.1",
        "pandas>=1.5.3",
    ],
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Operating System :: Unix"],
      package_data={
          'gammabayes':['package_data/*.gz', 
                        'package_data/*.txt',
                        'package_data/*.fits',
                        'BFCalc/griddata/*.npy',
                        'BFCalc/220913_145357/bfmlambdaarray.npy',
                        'BFCalc/220913_145357/lambdarange.npy',
                        'BFCalc/220913_145357/massrange.npy',
                        'BFCalc/dark_matter_spectra/AtProduction_gammas.dat',
                        'BFCalc/darkSUSY_BFs/darkSUSY_BFs_cleaned.csv',
                        'BFCalc/temp/*',
                        ]
      }
      )
