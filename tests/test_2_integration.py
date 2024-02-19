from gammabayes.utils.integration import (
    construct_log_dx,
    logspace_riemann, 
    iterate_logspace_integration
)
import numpy as np

def test_construct_log_dx():
    x = np.linspace(-1,1,2001)
    logdxs = construct_log_dx(x)
    assert np.ptp(logdxs)<1e-9


def test_logspace_riemann():
    intervals = 10000
    x = np.linspace(0,1,intervals)
    logy = x*0
    integration_result = logspace_riemann(logy=logy, x=x)

    # Real low bar for entry. In exact-ness comes from the need for 
        # linear binning and slightly weird integration for the 
        # inverse transform sampling
    assert np.abs(integration_result)<10/intervals

def test_iterate_logspace_riemann():
    intervals           = 1000

    x1                  = np.linspace(0,1,intervals)
    x2                  = np.linspace(0,1,intervals)

    logy                = 0*np.meshgrid(x1,x2,indexing='ij')[0]

    integration_result  = iterate_logspace_integration(logy=logy, 
                axes = [x1,x2], 
                logspace_integrator = logspace_riemann)

    # Real low bar for entry
    assert np.abs(integration_result)<10/intervals

