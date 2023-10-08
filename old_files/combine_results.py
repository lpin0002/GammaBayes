
import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm
import sys, os, yaml
from gammabayes.utils.utils import read_config_file
from gammabayes.hyperparameter_likelihood import hyperparameter_likelihood
import pickle





inputs = read_config_file(sys.argv[1])


currentdirectory = os.getcwd()
stemdirectory = f"{currentdirectory}/data/{inputs['identifier']}/singlerundata"
print("\nstem directory: ", stemdirectory, '\n')

rundirs = [x[0] for x in os.walk(stemdirectory)][1:]

hyperparameter_likelihood_instance = hyperparameter_likelihood()


with open(rundirs[0]+'/hyper_parameter_data.pkl', 'rb') as pickle_file:
        hyper_parameter_data = pickle.load(pickle_file)

# Now, loaded_data contains the object with NumPy arrays
print("Loaded data:", hyper_parameter_data.keys())


hyperparameter_likelihood_instance.initiate_from_dict(hyper_parameter_data)


logmassrange = hyperparameter_likelihood_instance.hyperparameter_axes_tuple[0][0]

for rundir in rundirs[1:]:
    try:
        # Open and read the JSON file
        with open(rundir+'/hyper_parameter_data.pkl', 'rb') as pickle_file:
            loaded_data = pickle.load(pickle_file)

        hyperparameter_likelihood_instance.add_results(loaded_data["log_margresults"])

    except Exception as e:
        print("Error:", str(e))



xi_windowwidth      = 50/np.sqrt(inputs['totalevents'])


xi_lowerbound       = inputs['xi']-xi_windowwidth
xi_upperbound       = inputs['xi']+xi_windowwidth



if xi_lowerbound<0:
    xi_lowerbound = 0
if xi_upperbound>1:
    xi_upperbound = 1


xi_range            = np.linspace(xi_lowerbound, xi_upperbound, inputs['nbins_xi']) 

     

hyperparameter_likelihood_instance.create_mixture_log_posterior(mixture_axes=(xi_range, 1-xi_range))
log_posterior = np.squeeze(hyperparameter_likelihood_instance.unnormed_log_posterior)

import matplotlib.pyplot as plt

plt.figure()
plt.pcolormesh(log_posterior)
plt.show(block=True)




hyperparameter_likelihood_instance.save_data(directory_path=f"{currentdirectory}/data/{inputs['identifier']}")


print(f"inputs['identifier']: {inputs['identifier']}")
hyperparameter_likelihood_instance.plot_posterior(identifier=inputs['identifier'], Nevents=inputs['totalevents'], 
                                                  truevals=(inputs['xi'], inputs['logmass']))

log_posterior = np.squeeze(hyperparameter_likelihood_instance.unnormed_log_posterior)

np.save(f"data/{inputs['identifier']}/log_posterior", log_posterior)
np.save(f"data/{inputs['identifier']}/xi_range", xi_range)
np.save(f"data/{inputs['identifier']}/logmassrange", logmassrange)