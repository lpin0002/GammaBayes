
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
    
print(config_file_path)
print(config_inputs)
try:
    common_norm_matrices = config_inputs['common_norm_matrices']
except:
    common_norm_matrices = False

if common_norm_matrices:

    stemdirname = time.strftime(f"data/{config_inputs['stem_identifier']}")

    workingfolder = os.getcwd()

    try:
        os.mkdir(f"{workingfolder}/{stemdirname}")
    except:
        raise Exception("Stem folder already exists")

    matgen_config_file_path = f"data/{config_inputs['stem_identifier']}/irfmat_gen_config.yaml"
    save_config_file(config_inputs, matgen_config_file_path)

    config_inputs['jobname'] = f'MatrixGen_{config_inputs['jobname']}'
    config_inputs['numcores'] = 1
    config_inputs['mem_per_cpu'] = config_inputs['mem_for_matrix_gen']
    config_inputs['time_hrs'] = config_inputs['time_for_matrix_gen_hrs']
    config_inputs['time_mins'] = config_inputs['time_for_matrix_gen_mins']

    matgen_job_str = makejobscripts(config_inputs, matgen_config_file_path, path_to_run_file='-m gammabayes.utils.ozstar.gen_norm_matrices')

    matgen_jobscript_filename = f"{stemdirname}/matgen_jobscript.sh"

    with open(matgen_jobscript_filename, 'w') as f:
        f.write(matgen_job_str)

    os.system(f"sbatch {matgen_jobscript_filename}")

else:
    gen_scripts(config_inputs)
