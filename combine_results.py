
import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm
import sys, os, yaml
from gammabayes.utils.utils import read_config_file, load_hyperparameter_pickle
from gammabayes.hyper_inference.hyperparameter_likelihood import hyperparameter_likelihood






config_inputs = read_config_file(sys.argv[1])

hyperparameter_likelihood_instance = hyperparameter_likelihood()

currentdirectory = os.getcwd()
stemfolder = f"{currentdirectory}/data/{config_inputs['identifier']}"
stemdatadirectory = f"{stemfolder}/singlerundata"

print("\nstem directory: ", stemfolder, '\n')



rundirs = [x[0] for x in os.walk(stemdatadirectory)][1:]

init_hyperparameter_likelihood_instance_pickle = load_hyperparameter_pickle(rundirs[0]+'/hyper_parameter_data.pkl')

hyperparameter_likelihood_instance.init_from_pickle(init_hyperparameter_likelihood_instance_pickle)




lambdawindowwidth      = 9/np.sqrt(config_inputs['totalevents'])


lambdalowerbound       = config_inputs['xi']-lambdawindowwidth
lambdaupperbound       = config_inputs['xi']+lambdawindowwidth




if lambdalowerbound<0:
    lambdalowerbound = 0
if lambdaupperbound>1:
    lambdaupperbound = 1


xi_range            = np.linspace(lambdalowerbound, lambdaupperbound, config_inputs['nbins_xi']) 



for rundir in rundirs:
    try:
        rundir_hyperparameter_data_pickle = load_hyperparameter_pickle(rundir+'/hyper_parameter_data.pkl')
        rundir_log_hyper_like = hyperparameter_likelihood_instance.create_mixture_log_hyperparameter_likelihood(mixture_axes=(xi_range, 1-xi_range), log_marg_results=rundir_hyperparameter_data_pickle['log_marg_results'])
        hyperparameter_likelihood_instance.combine_hyperparameter_likelihoods(rundir_log_hyper_like)
    except:
        pass

print(hyperparameter_likelihood_instance.mixture_axes[0])

hyperparameter_likelihood_instance.save_data(directory_path=stemfolder)



hyperparameter_likelihood_instance.plot_posterior(config_file=config_inputs, mixture_axes=hyperparameter_likelihood_instance.mixture_axes[0])


np.save(f"data/{config_inputs['identifier']}/log_posterior", hyperparameter_likelihood_instance.log_posterior)
np.save(f"data/{config_inputs['identifier']}/xi_range", xi_range)
np.save(f"data/{config_inputs['identifier']}/logmassrange", stemfolder)