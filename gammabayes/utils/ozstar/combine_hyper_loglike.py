import sys, os
from gammabayes.utils.config_utils import read_config_file
from gammabayes.utils import load_pickle
from gammabayes.hyper_inference import discrete_hyperparameter_likelihood
from scipy import special
from gammabayes.utils.plotting import logdensity_matrix_plot
import matplotlib.pyplot as plt

try:
    config_file_path = sys.argv[1]
except:
    warnings.warn('No configuration file given')
    config_file_path = os.path.dirname(__file__)+'/Z2_DM_3COMP_BKG_config_default.yaml'


# Extracting base config dict

config_dict = read_config_file(config_file_path)


# Extracting relevant file paths

workingfolder = os.getcwd()

stemdirname = f"data/{config_dict['stem_identifier']}"


# File path that contains all the folders containing individual 'job' results
rundata_filepath = f"{workingfolder}/{stemdirname}/rundata"



# Get a list of all files and directories in rundata_filepath
all_items = os.listdir(rundata_filepath)

# Filter out only directories within rundata_filepath
subdirectories = [os.path.join(rundata_filepath, item) for item in all_items if os.path.isdir(os.path.join(rundata_filepath, item))]

initial_subdirectory = subdirectories[0]
initial_saved_data = load_pickle(f"{initial_subdirectory}/run_data.pkl")

hyper_class_instance = discrete_hyperparameter_likelihood(axes=initial_saved_data['axes'],
                 dependent_axes=initial_saved_data['dependent_axes'],
                 hyperparameter_axes=initial_saved_data['hyperparameter_axes'],
                 mixture_axes=initial_saved_data['mixture_axes'],
                 log_hyperparameter_likelihood=initial_saved_data['log_hyperparameter_likelihood'],)



for subdirectory in subdirectories[1:]:
    saved_data = load_pickle(subdirectory+'/run_data.pkl')
    hyper_class_instance.update_hyperparameter_likelihood(saved_data['log_hyperparameter_likelihood'])


plot_log_hyper_param_likelihood = hyper_class_instance.log_hyperparameter_likelihood-special.logsumexp(hyper_class_instance.log_hyperparameter_likelihood)

try:
    fig, ax = logdensity_matrix_plot([*hyper_class_instance.mixture_axes, initial_saved_data['massrange'], ], plot_log_hyper_param_likelihood, 
                                    truevals=[saved_data['config']['signalfraction'], 
                                                saved_data['config']['ccr_of_bkg_fraction'], 
                                                saved_data['config']['diffuse_of_astro_fraction'], 
                                                saved_data['config']['dark_matter_mass'],],   
                                    sigmalines_1d=1, contours2d=1, plot_density=1, single_dim_yscales='linear',
                                    axis_names=['sig/total', 'ccr/bkg', 'diffuse/astro', 'mass [TeV]',], 
                                    suptitle=float(saved_data['config']['numjobs'])*float(saved_data['config']['Nevents_per_job']), figsize=(10,10))

except:
    fig, ax = logdensity_matrix_plot([*hyper_class_instance.mixture_axes, initial_saved_data['massrange'], ], plot_log_hyper_param_likelihood, 
                                    truevals=[saved_data['config']['signalfraction'], 
                                                saved_data['config']['ccr_of_bkg_fraction'], 
                                                saved_data['config']['diffuse_of_astro_fraction'], 
                                                saved_data['config']['dark_matter_mass'],],   
                                    sigmalines_1d=0, contours2d=0, plot_density=0, single_dim_yscales='linear',
                                    axis_names=['sig/total', 'ccr/bkg', 'diffuse/astro', 'mass [TeV]',], 
                                    suptitle=float(saved_data['config']['numjobs'])*float(saved_data['config']['Nevents_per_job']), figsize=(10,10))
fig.figure.dpi = 120
ax[3,3].set_xscale('log')
ax[3,0].set_yscale('log')
ax[3,1].set_yscale('log')
ax[3,2].set_yscale('log')
plt.tight_layout()
plt.savefig(f"{stemdirname}/hyper_loglike_scan_corner.pdf")
plt.show()
