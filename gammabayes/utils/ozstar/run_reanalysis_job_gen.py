
from gammabayes.utils.config_utils import read_config_file, save_config_file
from gammabayes.utils.ozstar.ozstar_job_script_gen import gen_scripts
from gammabayes.utils.ozstar.make_ozstar_scripts import makejobscripts
import sys, os, time
from warnings import warn



try:
    config_file_path = sys.argv[1]
    config_inputs = read_config_file(config_file_path)


except KeyboardInterrupt:
    raise KeyboardInterrupt

except:
    config_file_path = os.path.dirname(__file__)+'/default_ozstar_config.yaml'
    config_inputs = read_config_file(config_file_path)
    
config_inputs['reanalysis'] = True

if 'reanalysis_stem' not in config_inputs:
    config_inputs['reanalysis_stem'] = 'reanalysis'

if 'reanalysis_result_save_filename' not in config_inputs:
    config_inputs['result_save_filename'] = config_inputs['reanalysis_stem']+'_results.h5'
else:
    config_inputs['result_save_filename'] = config_inputs['reanalysis_result_save_filename']

if 'reanalysis_posterior_samples_file_name' not in config_inputs:
    config_inputs['posterior_samples_file_name'] = config_inputs['reanalysis_stem']+'_posterior_samples.h5'
else:
    config_inputs['posterior_samples_file_name'] = config_inputs['reanalysis_posterior_samples_file_name']


stem_identifier = config_inputs['stem_identifier']
log_psf_norm_matrix_path = f"data/{stem_identifier}/log_psf_normalisations.npy"
log_edisp_norm_matrix_path = f"data/{stem_identifier}/log_edisp_normalisations.npy"



config_inputs['log_psf_norm_matrix_path']   = log_psf_norm_matrix_path
config_inputs['log_edisp_norm_matrix_path'] = log_edisp_norm_matrix_path



if 'reanalysis_full_results_filename' not in config_inputs:
    config_inputs['full_results_filename'] = config_inputs['reanalysis_stem']+'_full_results.h5'

else:
    config_inputs['full_results_filename'] = config_inputs['reanalysis_full_results_filename']



print(config_file_path)
print(config_inputs)


# Presuming that the run has already been done, so the common IRF normalisation matrices
    # is set to be common across the jobs, then they should have already been generated
    # anyway. So they are not re-generated here.


gen_scripts(config_inputs)
