
from gammabayes.utils.config_utils import read_config_file, save_config_file
from gammabayes.utils import generate_unique_int_from_string
from gammabayes.utils.ozstar.make_ozstar_scripts import makejobscripts
import sys, os, time
from tqdm import tqdm
from warnings import warn






def gen_scripts(config_inputs): 
    # workingfolder = os.path.realpath(os.path.join(sys.path[0]))
    workingfolder = os.getcwd()
    print('\n\n\nWorking directory: '+workingfolder,'\n\n\n')


    os.makedirs(workingfolder+"/data", exist_ok=True)
    os.makedirs(workingfolder+"/data/LatestFolder", exist_ok=True)

        
    if config_inputs['stem_identifier'] is None:
        config_inputs['stem_identifier'] =  time.strftime("%m%d%H")

    stemdirname = time.strftime(f"data/{config_inputs['stem_identifier']}")

    print('\n\n\nStem directory: '+stemdirname,'\n\n\n')
    os.makedirs(f"{workingfolder}/{stemdirname}", exist_ok=True)

    full_config_file_path = f"data/{config_inputs['stem_identifier']}/config.yaml"
    save_config_file(config_inputs, full_config_file_path)
    print('\n\n\nSaved Original Config File.\n\n\n')

    rundata_filepath = f"{workingfolder}/{stemdirname}/rundata"
    print('\n\n\nCreated Run Directory: '+rundata_filepath,'\n\n\n')

    os.makedirs(rundata_filepath, exist_ok=True)

    print("Initial folder setup complete.")

    for runnum in tqdm(range(1,config_inputs['numjobs']+1), desc='Generating (and/or executing) bash jobs'):
        job_config_inputs = config_inputs
        job_config_inputs['run_number'] = runnum

        run_data_folder = f"{workingfolder}/{stemdirname}/rundata/{runnum}"

        os.makedirs(run_data_folder)

        job_config_inputs['save_path'] = f"{run_data_folder}/"
        job_config_inputs['Nevents'] = job_config_inputs['Nevents_per_job']
        job_config_inputs['jobname'] = f"{job_config_inputs['stem_identifier']}_{runnum}"

        job_seed = generate_unique_int_from_string(job_config_inputs['jobname'])

        job_config_inputs['seed'] = int(job_seed)

        job_config_file_name = f"{run_data_folder}/config.yaml"
        save_config_file(job_config_inputs, job_config_file_name)

        bash_file_body = makejobscripts(job_config_inputs, job_config_file_name, path_to_run_file='-m gammabayes.standard_inference.run_standard_inf')

        jobscript_filename = f"{run_data_folder}/jobscript.sh"

        with open(jobscript_filename, 'w') as f:
            f.write(bash_file_body)

        if bool(config_inputs['immediate_run']):
            os.system(f"sbatch {jobscript_filename}")

        
            
if __name__=="__main__":
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

    gen_scripts(config_inputs)
