
from gammabayes.utils.config_utils import read_config_file, save_config_file
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
    print(config_file_path)
    config_inputs = read_config_file(config_file_path)
    print(config_inputs)

    bash_file_body = makejobscripts(config_inputs, config_file_path, path_to_run_file='gammabayes.standard_inference.Z3_DM_3COMP_BKG')



    raise Exception
    # workingfolder = os.path.realpath(os.path.join(sys.path[0]))
    workingfolder = os.getcwd()
    print('\n\n\nWorking directory: '+workingfolder,'\n\n\n')
    

    try:
        os.mkdir(workingfolder+"/data")
    except:
        print("data folder already exists")

    try:
        os.mkdir(workingfolder+"/data/LatestFolder")
    except:
        print("LatestFolder folder already exists")
        
    if config_inputs['stem_identifier'] is None:
        config_inputs['stem_identifier'] =  time.strftime("%m%d%H")
    
    stemdirname = time.strftime(f"data/{config_inputs['stem_identifier']}")


    try:
        os.mkdir(f"{workingfolder}/{stemdirname}")
    except:
        raise Exception("Stem folder already exists")
    
    os.makedirs(f"{workingfolder}/{stemdirname}/singlerundata", exist_ok=True)

    for runnum in range(1,config_inputs['numjobs']+1):
        single_run_data_folder = f"{workingfolder}/{stemdirname}/singlerundata/{runnum}"
        
        os.makedirs(single_run_data_folder)



    save_config_file(config_inputs, f"{single_run_data_folder}/config.yaml")

    with open(f"{single_run_data_folder}/jobscript.sh", 'w') as f:
        f.write(bash_file_body)
    
    os.system(f"sbatch {single_run_data_folder}/jobscript.sh")


