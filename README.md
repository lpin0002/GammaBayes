# GammaBayes
__Author(s)__: Liam Pinchbeck (Liam.Pinchbeck@monash.edu)

__Supervisor(s)__: Csaba Balazs, Eric Thrane

__Documentation__: [ReadtheDocs](https://gammabayes.readthedocs.io/en/latest/index.html)

## Referencing

This code is an adaption of a previous work by Abhi Mangipudi and the relevant paper for referencing can be found [here](https://arxiv.org/abs/2112.10371) from ArXiv. The bibtex code for it is

@article{Mangipudi:2021ivm,
    author = "Mangipudi, Abhi and Thrane, Eric and Balazs, Csaba",
    title = "{Bayesian WIMP detection with the Cherenkov Telescope Array}",
    eprint = "2112.10371",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    month = "12",
    year = "2021"
}


## Warning

Within the analysis we slice into matrices for the normalisation values of likelihood functions to enforce a normalisation on the interpolation done.
These matrices can be quite large depending on the resolution of the axes chosen. Keep this in mind when implementing multi-processing as pytohn will
duplicate the arrays instead of reference the same one.

## Introduction

This coding repository contains a Bayesian Inference pipeline for calculating dark matter related observables from (simulated) observations from the galactic centre. Example files that run the simulation and analysis can be found within the `docs` folder. All documentation for the code is within the notebook files contained within that folder including all the components that make up the analysis in the oncoming publication.

A python package version of the code exists on `PyPi` that can be installed with the command,

`pip install gammabayes`.

This will also take care of the required dependencies for the project.
