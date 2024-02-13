import warnings




import os, sys, time, random
from tqdm import tqdm

from gammabayes.likelihoods.irfs import IRF_LogLikelihood
from gammabayes.utils import update_with_defaults, logspace_riemann, hdp_credible_interval_1d

from gammabayes.hyper_inference import DiscreteAdaptiveScan as discrete_hyperparameter_likelihood
from gammabayes.priors import DiscreteLogPrior

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

def test_gauss_sim():
    warnings.simplefilter("ignore", category=UserWarning)
    warnings.simplefilter("ignore", category=DeprecationWarning)
    warnings.simplefilter("ignore", category=RuntimeWarning)

    random.seed(1)

    energy_true_axis, longitudeaxistrue, latitudeaxistrue = np.logspace(-1,2,31), np.linspace(-5,5,21), np.linspace(-4,4,17)

    energy_recon_axis, longitudeaxis, latitudeaxis = np.logspace(-1,2,16), np.linspace(-5,5,11), np.linspace(-4,4,9)

    irf_loglike = IRF_LogLikelihood(axes=[energy_recon_axis, longitudeaxis, latitudeaxis], dependent_axes=[energy_true_axis, longitudeaxistrue, latitudeaxistrue])
    log_psf_normalisations, log_edisp_normalisations = irf_loglike.create_log_norm_matrices()

    NumEvents           = int(5e2)
    true_centre         = 20.0
    std_dev             = 5.0
    g1_fraction         = 0.3

    ng1 = int(round(g1_fraction*NumEvents))
    ng2 = int(round((1-g1_fraction)*NumEvents))

    def _fake_func(energy, longitude, latitude, spectral_parameters, spatial_parameters={}):
        result = -(energy-spectral_parameters['centre'])**2/(2.*std_dev**2) - 0.5*np.log(2*np.pi*std_dev**2)
        return result

    gauss1_prior = DiscreteLogPrior(logfunction=_fake_func, 
                             name='Gauss1',
                             axes=[energy_true_axis, longitudeaxistrue, latitudeaxistrue], 
                             axes_names=['energy', 'lon', 'lat'],
                             default_spectral_parameters={'centre':true_centre,}, 
                              )
                              
    g1_energy_vals, g1_lonvals, g1_latvals  = gauss1_prior.sample(ng1)
    print('\n------------------------\n')
    gauss2_prior = DiscreteLogPrior(logfunction=_fake_func, 
                             name='Gauss2',
                             axes=[energy_true_axis, longitudeaxistrue, latitudeaxistrue], 
                             axes_names=['energy', 'lon', 'lat'],
                             default_spectral_parameters={'centre':5.0,}, 
                              )


    g2_energy_vals, g2_lonvals, g2_latvals  = gauss2_prior.sample(ng2)


    within_one_std_dev = np.sum((g1_energy_vals > true_centre - std_dev) & (g1_energy_vals < true_centre + std_dev))
    proportion_within_one_std_dev = within_one_std_dev / (g1_fraction*NumEvents)

    # Check if more than 20% of samples are within one standard deviation
    print(proportion_within_one_std_dev)
    assert proportion_within_one_std_dev > 0.5
    assert np.ptp(g1_energy_vals)>1e-1

    g1_energy_meas, g1_longitude_meas, g1_latitude_meas = np.asarray([irf_loglike.sample(dependentvalues=[*nuisance_vals]) for nuisance_vals in zip(g1_energy_vals,g1_lonvals,g1_latvals)]).T
    g2_energy_meas, g2_longitude_meas, g2_latitude_meas = np.asarray([irf_loglike.sample(dependentvalues=[*nuisance_vals]) for nuisance_vals in zip(g2_energy_vals,g2_lonvals,g2_latvals)]).T

    measured_energy = list(g1_energy_meas)+list(g2_energy_meas)
    measured_longitude = list(g1_longitude_meas)+list(g2_longitude_meas)
    measured_latitude = list(g1_latitude_meas)+list(g2_latitude_meas)

    centre_range = np.logspace(1.2, np.log10(30),61)

    hyperparameter_likelihood_instance = discrete_hyperparameter_likelihood(
        priors                  = (gauss1_prior, gauss2_prior), 
        likelihood              = irf_loglike, 
        dependent_axes          = (energy_true_axis,  longitudeaxistrue, latitudeaxistrue), 
        hyperparameter_axes     = [
            {'spectral_parameters'  : {'centre'   : centre_range}, 
                                        }, 
                                        ], 
        numcores                = 4, 
        likelihoodnormalisation = log_psf_normalisations+log_edisp_normalisations)

    measured_energy = [float(measured_energy_val) for measured_energy_val in measured_energy]
    margresults = hyperparameter_likelihood_instance.nuisance_log_marginalisation(
        axisvals= (measured_energy, measured_longitude, measured_latitude)
        )

    g1frac_range = np.linspace(0.1,0.5,51)
    new_log_posterior = hyperparameter_likelihood_instance.create_discrete_mixture_log_hyper_likelihood(
        mixture_axes=(g1frac_range,), log_margresults=margresults)
    new_log_posterior = np.squeeze(new_log_posterior - special.logsumexp(new_log_posterior))

    print('Going to calculate marginal dists now')

    # Checking whether the reconstructed values are within 2 sigma 
    #   for each marginal distribution
    log_g1frac_marginal_dist = logspace_riemann(logy=new_log_posterior, x=centre_range, axis=1)

    log_g1frac_marginal_dist = log_g1frac_marginal_dist - logspace_riemann(
        logy=log_g1frac_marginal_dist, x=g1frac_range)
    g1frac_bounds = hdp_credible_interval_1d(y=np.exp(log_g1frac_marginal_dist), x=g1frac_range, sigma=2)

    g1_truth_bool = (g1_fraction>=g1frac_bounds[0]) and (g1_fraction<=g1frac_bounds[1])
    print('g1_truth_bool: ', g1_truth_bool)
    assert g1_truth_bool


    log_centre_marginal_dist = logspace_riemann(logy=new_log_posterior, x=g1frac_range, axis=0)
    log_centre_marginal_dist = log_centre_marginal_dist - logspace_riemann(
        logy=log_centre_marginal_dist, x=centre_range)
    centre_bounds = hdp_credible_interval_1d(y=np.exp(log_centre_marginal_dist), x=centre_range, sigma=2)


    centre_truth_bool = (true_centre>=centre_bounds[0]) and (true_centre<=centre_bounds[1])
    print('centre_truth_bool: ', centre_truth_bool)
    assert centre_truth_bool


    warnings.simplefilter("default", category=UserWarning)
    warnings.simplefilter("default", category=DeprecationWarning)
    warnings.simplefilter("default", category=RuntimeWarning)
