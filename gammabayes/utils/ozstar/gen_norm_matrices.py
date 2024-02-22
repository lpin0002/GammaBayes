from gammabayes.utils.config_utils import (
    read_config_file, 
    save_config_file,
    create_true_axes_from_config, 
    create_recon_axes_from_config
)
from gammabayes.likelihoods.irfs import IRF_LogLikelihood

from gammabayes.utils.ozstar.ozstar_job_script_gen import gen_scripts
from gammabayes.utils.ozstar.make_ozstar_scripts import makejobscripts
import sys, os, time
from warnings import warn
import numpy as np
import copy

        
config_file_path = sys.argv[1]
config_inputs = read_config_file(config_file_path)
os.makedirs('data', exist_ok=True)
os.makedirs(f'data/{config_inputs["stem_identifier"]}')
    

energy_true_axis,  longitudeaxistrue, latitudeaxistrue       = create_true_axes_from_config(config_inputs)
energy_recon_axis, longitudeaxis,     latitudeaxis           = create_recon_axes_from_config(config_inputs)


irf_loglike = IRF_LogLikelihood(axes   =   [energy_recon_axis,    longitudeaxis,     latitudeaxis], 
                                dependent_axes =   [energy_true_axis,     longitudeaxistrue, latitudeaxistrue])

log_psf_normalisations, log_edisp_normalisations = irf_loglike.create_log_norm_matrices()

print("Log of normalisations of IRFs generated.")

stemdirname = f"data/{config_inputs['stem_identifier']}"

log_psf_norm_matrix_path = f"{stemdirname}/log_psf_normalisations.npy"
log_edisp_norm_matrix_path = f"{stemdirname}/log_edisp_normalisations.npy"



np.save(log_psf_norm_matrix_path, log_psf_normalisations)
np.save(log_edisp_norm_matrix_path, log_edisp_normalisations)
print("Saved log of normalisations of IRFs.")


config_inputs['log_psf_norm_matrix_path']   = log_psf_norm_matrix_path
config_inputs['log_edisp_norm_matrix_path'] = log_edisp_norm_matrix_path


matgen_config_file_path = f"data/{config_inputs['stem_identifier']}/irfmat_gen_config.yaml"
save_config_file(config_inputs, matgen_config_file_path)


print("Generating combine job bash script...")
combine_config_inputs = copy.deepcopy(config_inputs)

if 'combine_time_mins' in combine_config_inputs:
    combine_config_inputs['time_mins'] = combine_config_inputs['combine_time_mins']

if 'combine_time_hrs' in combine_config_inputs:
    combine_config_inputs['time_hrs'] = combine_config_inputs['combine_time_hrs']



if 'combine_numcores' in combine_config_inputs:
    combine_config_inputs['numcores'] = combine_config_inputs['combine_numcores']
elif 'numcores' in combine_config_inputs:
    combine_config_inputs['numcores'] = combine_config_inputs['numcores']
else:
    combine_config_inputs['numcores'] = 8

if 'combine_mem_per_cpu' in combine_config_inputs:
    combine_config_inputs['mem_per_cpu'] = combine_config_inputs['combine_mem_per_cpu']
elif 'mem_per_cpu' in combine_config_inputs:
    combine_config_inputs['mem_per_cpu'] = combine_config_inputs['mem_per_cpu']
else:
    combine_config_inputs['mem_per_cpu'] = 1000



combine_config_inputs['jobname'] = f"{config_inputs['stem_identifier']}_combine"

combine_config_file_path = f"data/{config_inputs['stem_identifier']}/combine_config.yaml"


save_config_file(combine_config_inputs, combine_config_file_path)

combine_bash_file_body = makejobscripts(combine_config_inputs, combine_config_file_path, path_to_run_file='-m gammabayes.utils.ozstar.combine_results')


with open(f"data/{config_inputs['stem_identifier']}/combine_jobscript.sh", 'w') as f:
    f.write(combine_bash_file_body)


print("Scheduling the main job scripts...")
gen_scripts(config_inputs)




