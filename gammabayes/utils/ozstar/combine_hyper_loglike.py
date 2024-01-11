import sys, os
from gammabayes.utils.config_utils import read_config_file, create_true_axes_from_config, create_recon_axes_from_config
from gammabayes.utils.import_utilities import dynamic_import
from gammabayes.likelihoods.irfs import irf_loglikelihood
from gammabayes.utils import load_pickle, extract_axes
from scipy import special
from gammabayes.utils.plotting import logdensity_matrix_plot
import matplotlib.pyplot as plt
from tqdm import tqdm


config_file_path = sys.argv[1]


# Extracting base config dict

config_dict = read_config_file(config_file_path)
discrete_hyperparameter_likelihood = dynamic_import('gammabayes.hyper_inference', config_dict['hyper_parameter_scan_class'])

energy_true_axis,  longitudeaxistrue, latitudeaxistrue       = create_true_axes_from_config(config_dict)
energy_recon_axis, longitudeaxis,     latitudeaxis           = create_recon_axes_from_config(config_dict)


irf_loglike = irf_loglikelihood(axes   =   [energy_recon_axis,    longitudeaxis,     latitudeaxis], 
                                dependent_axes =   [energy_true_axis,     longitudeaxistrue, latitudeaxistrue])

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
                 likelihood=irf_loglike,
                 hyperparameter_axes=initial_saved_data['hyperparameter_axes'],
                 mixture_axes=initial_saved_data['mixture_axes'],
                 log_hyperparameter_likelihood=initial_saved_data['log_hyperparameter_likelihood'],)



for subdirectory in tqdm(subdirectories[1:], desc='Accessing directories and adding data'):
    saved_data = load_pickle(subdirectory+'/run_data.pkl')
    hyper_class_instance.update_hyperparameter_likelihood(saved_data['log_hyperparameter_likelihood'])


plot_log_hyper_param_likelihood = hyper_class_instance.log_hyperparameter_likelihood-special.logsumexp(hyper_class_instance.log_hyperparameter_likelihood)

try:
    print("Plotting corner density plot")

    fig, ax = logdensity_matrix_plot([*initial_saved_data['mixture_axes'], 
                                      *list(extract_axes(initial_saved_data['source_hyperparameter_input']).values())[::-1], ], 
                                      plot_log_hyper_param_likelihood, 
                                    truevals=[initial_saved_data['config']['signalfraction'], 
                                            initial_saved_data['config']['ccr_of_bkg_fraction'], 
                                            initial_saved_data['config']['diffuse_of_astro_fraction'], 
                                            initial_saved_data['config']['dark_matter_mass'],
                                            0.17],   
                                    sigmalines_1d=1, contours2d=1, plot_density=1, single_dim_yscales='linear',
                                    axis_names=[
                                        *initial_saved_data['config']['mixture_fraction_specifications'].keys(), 
                                        *list(extract_axes(initial_saved_data['source_hyperparameter_input']).keys())[::-1],], 
                                    suptitle=initial_saved_data['config']['Nevents'],
                                    figsize=(12,10))
    fig.figure.dpi = 120
    ax[3,3].set_xscale('log')
    ax[3,0].set_yscale('log')
    ax[3,1].set_yscale('log')
    ax[3,2].set_yscale('log')
    plt.tight_layout()
    print(stemdirname+'/hyper_loglike_corner.pdf')
    plt.savefig(stemdirname+'/hyper_loglike_corner.pdf')
    plt.show()

except:
    print("Failed plotting corner density plot")
    print("Plotting unnormalised corner plot")

    print(plot_log_hyper_param_likelihood.shape)
    print([axis for axis in extract_axes(initial_saved_data['source_hyperparameter_input']).values()])

    fig, ax = logdensity_matrix_plot([*initial_saved_data['mixture_axes'], 
                                      *list(extract_axes(initial_saved_data['source_hyperparameter_input']).values())[::-1], ], 
                                      plot_log_hyper_param_likelihood, 
                                      truevals=[initial_saved_data['config']['signalfraction'], 
                                                initial_saved_data['config']['ccr_of_bkg_fraction'], 
                                                initial_saved_data['config']['diffuse_of_astro_fraction'], 
                                                initial_saved_data['config']['dark_matter_mass'],
                                                0.17],   
                                                sigmalines_1d=0, contours2d=0, plot_density=1, single_dim_yscales='linear',
                                                axis_names=[
                                                    *initial_saved_data['config']['mixture_fraction_specifications'].keys(), 
                                                    *list(extract_axes(initial_saved_data['source_hyperparameter_input']).keys())[::-1],], 
                                                    suptitle=initial_saved_data['config']['Nevents'],
                                    figsize=(12,10))
    fig.figure.dpi = 120
    ax[3,3].set_xscale('log')
    ax[3,0].set_yscale('log')
    ax[3,1].set_yscale('log')
    ax[3,2].set_yscale('log')
    plt.tight_layout()
    print(stemdirname+'/hyper_loglike_corner.pdf')
    plt.savefig(stemdirname+'/hyper_loglike_corner.pdf')
    plt.show()