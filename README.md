# GammaBayes
__Author(s)__: Liam Pinchbeck (Liam.Pinchbeck@monash.edu)

__Supervisor(s)__: Csaba Balazs, Eric Thrane

__Documentation__: [ReadtheDocs](https://gammabayes.readthedocs.io/en/latest/index.html)

## Warning

Within the analysis we slice into matrices for the normalisation values of likelihood functions to enforce a normalisation on the interpolation done.
These matrices can be quite large depending on the resolution of the axes chosen. Keep this in mind when implementing multi-processing as python will
duplicate the arrays instead of reference the same one.

## Introduction

This coding repository contains a Bayesian Inference pipeline for calculating dark matter related observables from (simulated) observations from the galactic centre. Example files that run the simulation and analysis can be found within the `docs` folder. All documentation for the code is within the notebook files contained within that folder, that make up the [ReadTheDocs](https://gammabayes.readthedocs.io/en/latest/index.html) page and all the major components that make up the analysis in the oncoming publication.

A python package version of the code exists on `PyPi` that can be installed with the command,

`pip install gammabayes`.

This will also take care of the required dependencies for the project.
