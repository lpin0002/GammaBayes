
from gammabayes.utils.config_utils import read_config_file, save_config_file
from gammabayes.utils import generate_unique_int_from_string
from gammabayes.utils.ozstar.make_ozstar_scripts import makejobscripts
import sys, os, time, copy
from tqdm import tqdm
from warnings import warn






def gen_scripts(config_inputs): 

    
    # workingfolder = os.path.realpath(os.path.join(sys.path[0]))

    if config_inputs['stem_identifier'] is None:
        config_inputs['stem_identifier'] =  time.strftime("%m%d%H")

    stem_identifier = config_inputs['stem_identifier']
    stemdirname = time.strftime(f"data/{stem_identifier}")

    print('\n\n\nStem directory: '+stemdirname,'\n\n\n')


    workingfolder = os.getcwd()
    print('\n\n\nWorking directory: '+workingfolder,'\n\n\n')


    os.makedirs(workingfolder+"/data", exist_ok=True)
    os.makedirs(workingfolder+"/data/LatestFolder", exist_ok=True)

        


    os.makedirs(f"{workingfolder}/{stemdirname}", exist_ok=True)




    if 'reanalysis' in config_inputs:
        reanalysis = config_inputs['reanalysis']

        if 'reanalysis_stem' in config_inputs:
            file_stem = config_inputs['reanalysis_stem']

    else:
        reanalysis = False
        file_stem = stem_identifier

    full_config_file_path = f"data/{stem_identifier}/{file_stem}_config.yaml"
    save_config_file(config_inputs, full_config_file_path)
    print('\n\n\nSaved Original Config File.\n\n\n')



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


    if 'reanalysis' in combine_config_inputs:
        reanalysis = combine_config_inputs['reanalysis']
    else:
        reanalysis = False


    if reanalysis:
        reanalysis_stem = config_inputs['reanalysis_stem']

        combine_config_inputs['jobname'] = f"{reanalysis_stem}_combine"

        combine_config_file_path = f"data/{config_inputs['stem_identifier']}/{reanalysis_stem}_combine_config.yaml"


        save_config_file(combine_config_inputs, combine_config_file_path)

        combine_bash_file_body = makejobscripts(combine_config_inputs, combine_config_file_path, path_to_run_file='-m gammabayes.utils.ozstar.combine_results')


        with open(f"data/{config_inputs['stem_identifier']}/{reanalysis_stem}_combine_jobscript.sh", 'w') as f:
            f.write(combine_bash_file_body)
            
            
    else:
        combine_config_inputs['jobname'] = f"{config_inputs['stem_identifier']}_combine"

        combine_config_file_path = f"data/{config_inputs['stem_identifier']}/combine_config.yaml"


        save_config_file(combine_config_inputs, combine_config_file_path)

        combine_bash_file_body = makejobscripts(combine_config_inputs, combine_config_file_path, path_to_run_file='-m gammabayes.utils.ozstar.combine_results')


        with open(f"data/{config_inputs['stem_identifier']}/combine_jobscript.sh", 'w') as f:
            f.write(combine_bash_file_body)


        print("Scheduling the main job scripts...")



    rundata_filepath = f"{workingfolder}/{stemdirname}/rundata"
    print('\n\n\nCreated Run Directory: '+rundata_filepath,'\n\n\n')

    os.makedirs(rundata_filepath, exist_ok=True)

    print("Initial folder setup complete.")

    if 'eventdata_save_filename' in config_inputs:
        eventdata_save_filename = config_inputs['eventdata_save_filename']

        if not(eventdata_save_filename.endswith('.h5')):
            eventdata_save_filename = eventdata_save_filename+'.h5'
    else:
        eventdata_save_filename = 'event_data.h5'

    for runnum in tqdm(range(1,config_inputs['numjobs']+1), desc='Generating (and/or executing) bash jobs'):
        job_config_inputs = config_inputs
        job_config_inputs['run_number'] = runnum

        run_data_folder = f"{workingfolder}/{stemdirname}/rundata/{runnum}"

        os.makedirs(run_data_folder, exist_ok=True)

        job_config_inputs['save_path'] = f"{run_data_folder}/"
        job_config_inputs['Nevents'] = job_config_inputs['Nevents_per_job']
        job_config_inputs['jobname'] = f"{job_config_inputs['stem_identifier']}_{runnum}"


        path_to_measured_event_data = f"{run_data_folder}/{eventdata_save_filename}"
        if reanalysis:
            job_config_inputs['path_to_measured_event_data'] = path_to_measured_event_data

        job_config_inputs['save_path_for_measured_event_data'] = path_to_measured_event_data

        job_seed = generate_unique_int_from_string(job_config_inputs['jobname'])

        job_config_inputs['seed'] = int(job_seed)

        job_config_file_name = f"{run_data_folder}/{file_stem}_config.yaml"
        save_config_file(job_config_inputs, job_config_file_name)

        bash_file_body = makejobscripts(job_config_inputs, job_config_file_name, path_to_run_file='-m gammabayes.standard_inference.run_standard_inf')

        jobscript_filename = f"{run_data_folder}/{file_stem}_jobscript.sh"

        with open(jobscript_filename, 'w') as f:
            f.write(bash_file_body)

        if bool(config_inputs['immediate_run']):
            os.system(f"sbatch {jobscript_filename}")

        
            
if __name__=="__main__":

    config_file_path = sys.argv[1]
    config_inputs = read_config_file(config_file_path)

    print(config_file_path)
    print(config_inputs)

    gen_scripts(config_inputs)
