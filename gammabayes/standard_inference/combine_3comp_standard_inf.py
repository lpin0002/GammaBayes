import numpy as np, time, random, os, warnings, pickle, sys
from matplotlib import pyplot as plt
from tqdm import tqdm

from gammabayes import EventData, Parameter, ParameterSet
from gammabayes.likelihoods.irfs import IRF_LogLikelihood

from gammabayes.utils.config_utils import (
    read_config_file, 
    create_true_axes_from_config, 
    create_recon_axes_from_config, 
)

from gammabayes.utils import (
    logspace_riemann, 
    iterate_logspace_integration, 
    generate_unique_int_from_string, 
    dynamic_import
)
from gammabayes.dark_matter import CombineDMComps

from gammabayes.priors import DiscreteLogPrior, log_bkg_CCR_dist, TwoCompPrior
from gammabayes.priors.astro_sources import FermiGaggeroDiffusePrior, HESSCatalogueSources_Prior

from multiprocessing import Pool
from dynesty.pool import Pool as DyPool
from dynesty import NestedSampler


if __name__=="__main__":

    config_file_path = sys.argv[1]
    
    config_dict = read_config_file(config_file_path)

    assert (len(config_dict)>0) and (type(config_dict)==dict)

