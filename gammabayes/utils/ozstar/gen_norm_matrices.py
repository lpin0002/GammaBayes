from gammabayes.utils.config_utils import (
    read_config_file, 
    save_config_file,
    create_true_axes_from_config, 
    create_recon_axes_from_config
)
from gammabayes.likelihoods.irfs import irf_loglikelihood

from gammabayes.utils.ozstar.ozstar_job_script_gen import gen_scripts
import sys, os, time
from warnings import warn
import numpy as np

        
config_file_path = sys.argv[1]
config_inputs = read_config_file(config_file_path)
    

energy_true_axis,  longitudeaxistrue, latitudeaxistrue       = create_true_axes_from_config(config_inputs)
energy_recon_axis, longitudeaxis,     latitudeaxis           = create_recon_axes_from_config(config_inputs)


irf_loglike = irf_loglikelihood(axes   =   [energy_recon_axis,    longitudeaxis,     latitudeaxis], 
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

print("Scheduling the main job scripts")
gen_scripts(config_inputs)




