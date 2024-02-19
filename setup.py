from setuptools import setup, find_packages
import subprocess
import sys
import os
import time

setup(name='GammaBayes',
      description='A package for Bayesian dark matter inference',
      url='https://github.com/lpin0002/GammaBayes',
      author='Liam Pinchbeck',
      author_email='Liam.Pinchbeck@monash.edu',
      license="MIT",
      version='0.1.0',
      packages=find_packages(),

      # For a lot of the DM spectral classes we require that dict types are ordered
      python_requires='>=3.6',
      install_requires=[
          "astropy==5.3.4",
        "corner>=2.2.2",
        "dynesty==2.1.2",
        "jupyterlab>=3.6.3",
        "matplotlib>=3.7.1",
        "scipy==1.11.3",
        "tqdm>=4.65.0",
        "numpy>=1.23",
        "gammapy>=0.20.1",
        "pandas>=1.5.3",
        "pytest",
        "h5py",
    ],
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Operating System :: Unix",
],
      package_data={
          'gammabayes':['package_data/gll_iem_v06_gc.fits.gz', 
                        'package_data/hgps_catalog_v1.fits.gz',
                        'package_data/*.txt',
                        'package_data/*.fits',
                        'package_data/irf_fits_files/*',
                        'package_data/irf_fits_files/prod5/*.FITS.tar.gz',
                        'package_data/irf_fits_files/prod3b/*',
                        'dark_matter/channel_spectra/PPPC_Tables/*.dat',
                        'dark_matter/spectral_models/Z2_ScalarSinglet/annihilation_ratio_data/*.csv',
                        'dark_matter/spectral_models/Z2_ScalarSinglet/temp/*',
                        'standard_inference/*',
                        'utils/ozstar/*'
                        ]
      },
      )
