import sys, os, numpy as np, h5py
from gammabayes.utils.config_utils import read_config_file, create_true_axes_from_config, create_recon_axes_from_config
from gammabayes.utils.import_utilities import dynamic_import
from gammabayes.samplers.sampler_utils import ResultsWrapper
from gammabayes import ParameterSet
from scipy import special

import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from dynesty.pool import Pool as DyPool
from dynesty import NestedSampler


if __name__=="__main__":
    config_file_path = sys.argv[1]


    # Extracting base config dict

    config_dict = read_config_file(config_file_path)
    discrete_hyperparameter_likelihood = dynamic_import('gammabayes.hyper_inference', config_dict['hyper_parameter_scan_class'])

    prior_parameter_sets = [ParameterSet(set) for set in list(config_dict['prior_parameter_specifications'].items())]
    mixture_parameter_set = ParameterSet(config_dict['mixture_fraction_specifications'])
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
    initial_save_file_name = f"{initial_subdirectory}/results.h5"

    full_hyper_class_instance = discrete_hyperparameter_likelihood.load(file_name = initial_save_file_name)


    for subdirectory in tqdm(subdirectories[1:], desc='Accessing directories and adding data'):
        hyper_class_instance = discrete_hyperparameter_likelihood.load(file_name = f"{subdirectory}/results.h5", 
                                                                    overriding_class_input_dict={'prior_parameter_specifications':prior_parameter_sets})
        full_hyper_class_instance.add_log_nuisance_marg_results(hyper_class_instance.log_nuisance_marg_results)




    scan_type = config_dict['mixture_fraction_exploration_type']
    scan_type_sample = scan_type=='sample'

    full_hyper_class_instance.select_scan_output_exploration_class(
                log_nuisance_marg_results           = full_hyper_class_instance.log_nuisance_marg_results,
                mixture_parameter_specifications    = mixture_parameter_set,
                prior_parameter_specifications      = prior_parameter_sets,
                mixture_fraction_exploration_type   = scan_type
            )
        
    if 'combine_num_cores' in config_dict:
        combine_num_cores = config_dict['combine_num_cores']
    else:
        combine_num_cores = 1


    if scan_type_sample and combine_num_cores>1:

        with DyPool(njobs=combine_num_cores, loglike=full_hyper_class_instance.hyper_analysis_instance.ln_likelihood,
                    prior_transform=full_hyper_class_instance.hyper_analysis_instance.prior_transform) as pool:
            sampler = NestedSampler(pool.loglike, 
                                    pool.prior_transform, 
                                    pool=pool,
                                    ndim=full_hyper_class_instance.hyper_analysis_instance.ndim)
            
            sampler.run_nested()
            

            
            
        full_hyper_class_instance.hyper_analysis_instance.sampler = sampler
    else:
        full_hyper_class_instance.init_posterior_exploration()
        full_hyper_class_instance.run_posterior_exploration()

    full_hyper_class_instance.mixture_parameter_specifications = mixture_parameter_set
    full_hyper_class_instance.save_to_pickle(f"data/{config_dict['stem_identifier']}/full_results.pkl")
    
    sampler_results = full_hyper_class_instance.hyper_analysis_instance.sampler.results
    # Backup
    ResultsWrapper.save(file_name=f"data/{config_dict['stem_identifier']}/backup_posterior_samples.h5",sampler_results=sampler_results)



    from corner import corner
    from gammabayes.utils.plotting import defaults_kwargs
    from scipy.special import logsumexp
    defaults_kwargs['smooth'] = 2



    num_params = 4
    sampling_results = full_hyper_class_instance.hyper_analysis_instance.sampler.results.samples_equal()

    if ('plot_results_kwargs' in config_dict):
        plot_kwargs = config_dict['plot_results_kwargs']
        if 'save_fig_file_name' in plot_kwargs:
            save_fig_file_name = plot_kwargs['save_fig_file_name']

            save_fig_file_name = plot_kwargs.pop('save_fig_file_name')
        else:
            save_fig_file_name = 'full_result_corner.pdf'

        if 'figsize' in plot_kwargs:
            plot_kwargs['figsize'] = (int(val) for val in plot_kwargs['figsize'])

    else:
        save_fig_file_name = 'full_result_corner.pdf'
        plot_kwargs = {}


    fig = plt.figure()
    figure=corner(sampling_results, fig=fig,
        # labels=['sig/total', 'ccr/bkg', 'diffuse/astro bkg', r'$m_{\chi}$ [TeV]'],
        quantiles=[0.025, .16, 0.5, .84, 0.975],
        bins=[*[41]*config_dict['num_bkg_comp'],*[axis.size*2 for axis in prior_parameter_sets[0].axes]],
        #    range=([0.,0.2], [0.5,1.0], [0.,0.6], *[[axis.min(), axis.max()] for axis in prior_parameters.axes]),
        axes_scale=[*['linear']*config_dict['num_bkg_comp'], 'log',],
        
        **defaults_kwargs)

    for ax_idx, ax in enumerate(figure.get_axes()):
        # Find lines representing the quantiles (the 3rd line for each histogram is the median, based on the ordering in `quantiles`)
        lines = ax.get_lines()
        if (len(lines) > 2):
            if (ax_idx%(num_params+1)==0): 
                for line_idx, line in enumerate(lines): # Check if there are enough lines (for histograms)
                    if line_idx==2:
                        line.set_color('tab:green')  # Change the color of the median lines
                    elif line_idx<len(lines)-1:
                        line.set_color('tab:blue')
    plt.suptitle(str(float(config_dict['numjobs'])*float(config_dict['Nevents_per_job'])) + " events", size=24)



    plt.tight_layout()

    plt.savefig(f"data/{config_dict['stem_identifier']}/{save_fig_file_name}")
    plt.show()
