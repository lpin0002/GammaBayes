import os, sys, time
from gammabayes.priors import DiscreteLogPrior
import warnings
import numpy as np

def test_discrete_logprior():
    energy_true_axis, longitudeaxistrue, latitudeaxistrue = np.logspace(-1,2,16), np.linspace(-5,5,11), np.linspace(-4,4,9)

    def fake_func(energy, longitude, latitude, spectral_parameters, spatial_parameters={}):
        result = -(energy-spectral_parameters['centre'])**2/(2)
        return result

    warnings.simplefilter("ignore", category=UserWarning)

    discrete_logprior_instance = DiscreteLogPrior(logfunction=fake_func, 
                             name='Fake Test Func',
                             axes=[energy_true_axis, longitudeaxistrue, latitudeaxistrue], 
                             axes_names=['energy', 'lon', 'lat'],
                             default_spectral_parameters={'centre':1.0}, )
                             
    warnings.simplefilter("default", category=UserWarning)

    samples = discrete_logprior_instance.sample(100)
    print('can sample')
    log_prob_matrix = discrete_logprior_instance.construct_prior_array(spectral_parameters= {'centre':-0.5}, normalise=False)
    print('can construct')
    discrete_logprior_instance.normalisation(spectral_parameters= {'centre':0.0})
    print('can normalise')




