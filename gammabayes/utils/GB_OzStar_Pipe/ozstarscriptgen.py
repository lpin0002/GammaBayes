
import os, sys, time, math, yaml
parent_dir_name = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parent_dir_name)



from default_file_setup import default_file_setup
from config_utils import create_true_axes_from_config, create_recon_axes_from_config
from utils import makelogjacob, log_aeff, psf_test, edisp_test

def makejobscripts(logmass, xi_true, numberofruns, singlerunevents, numcores,
                   num_marg_hours, num_marg_minutes, num_combine_hours, num_combine_minutes,
                   nbins_logmass, nbins_xi, identifier = None, immediate_run=1,
                   marginalisation_mem = 200, combination_mem=1000, dmdensity_profile='einasto',
                   num_true_energy_bins_per_decade=300,
                   num_recon_energy_bins_per_decade=50,
                   true_spatial_res=0.2, recon_spatial_res=0.4):
    """An auxillary helper function for doing embarassingly parallel jobs on a 
    slurm job batch system.
    Args:
        logmass (float): Log_10 mass of the dark matter particle.

        xi_true (float): Fraction of events that originate from dark matter.

        numberofruns (int): Number of jobs to queue.

        singlerunevents (int): Number of events to simulate for a single job.

        numcores (int): Number of cores to use for a job.

        num_marg_hours: Number of hours to run the script that does the 
            nuisance marginalisation.

        num_marg_minutes: Number of minutes on top of the number of hours to
            run the script that does the nuisance marginalisation.

        num_combine_hours: Number of hours to run the script that combines the
            results of the nuisance marginalisation scripts.

        num_combine_minutes: Number of minutes on top of the number of hours to
            run the script that combines the results of the nuisance 
            marginalisation scripts.

        nbins_logmass: Number of log mass bins to evaluate.

        nbins_xi: Number of signal fraction bins to evaluate.

        identifier: String identifier for a run of jobs.

        immediate_run: Bool to immediately run the job scripts on creation.
            Default is True.

        marginalisation_mem: Amount of memory (in MB) to give each cores in a 
            single job doing the nuisance marginalisation.
            Default is 200.

        combination_mem: Amount of memory (in MB) to give each cores in a 
            single job combining the nuisance marginalisation results to get
            the posterior. Default is 1000.

        dmdensity_profile: string identifier for the dark matter mass density 
            distribution to use. Default is 'einasto'.
        """
    
    if int(num_marg_minutes)<10:
        num_marg_minutes = "10"
    else:
        num_marg_minutes = int(num_marg_minutes)
        
    if int(num_combine_minutes)<10:
        num_combine_minutes = "10"
    else:
        num_combine_minutes = int(num_combine_minutes)

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
        
    if identifier is None:
        identifier =  time.strftime("%m%d%H")
    
    stemdirname = time.strftime(f"data/{identifier}")


    try:
        os.mkdir(f"{workingfolder}/{stemdirname}")
    except:
        raise Exception("Stem folder already exists")
    run_data_folder = f"{workingfolder}/{stemdirname}/singlerundata"
    
    os.makedirs(run_data_folder, exist_ok=True)

    

    stem_config_dict = {
    'identifier'        : identifier,
    'Nevents'           : singlerunevents,
    'logmass'           : logmass,
    'xi'                : xi_true,
    'nbins_logmass'     : nbins_logmass,
    'nbins_xi'          : nbins_xi,
    'dmdensity_profile' : dmdensity_profile,
    'numcores'          : numcores,
    'totalevents'       : singlerunevents*numberofruns,
    'batch_job'         : 1,

    'log10_true_energy_min'   : -0.5,
    'log10_true_energy_max'   : 1.5,
    'log10_true_energy_bins_per_decade' : num_true_energy_bins_per_decade,
    'true_spatial_res'        : true_spatial_res,
    'true_longitude_min'      : -3.5,
    'true_longitude_max'      : 3.5,
    'true_latitude_min'       : -3.1,
    'true_latitude_max'       : 3.1,

    'log10_recon_energy_min'   : -0.5,
    'log10_recon_energy_max'   : 1.5,
    'log10_recon_energy_bins_per_decade' : num_recon_energy_bins_per_decade,
    'recon_spatial_res'        : recon_spatial_res,
    'recon_longitude_min'      : -3.5,
    'recon_longitude_max'      : 3.5,
    'recon_latitude_min'       : -3.1,
    'recon_latitude_max'       : 3.1,
}
    


    log10_eaxis_true, longitude_axis_true, latitude_axis_true = create_true_axes_from_config(stem_config_dict)


    log10_eaxis, longitude_axis, latitude_axis = create_recon_axes_from_config(stem_config_dict)
    default_file_setup(setup_irfnormalisations=1, setup_astrobkg=0,
                       log10eaxistrue=log10_eaxis_true, longitudeaxistrue=longitude_axis_true, latitudeaxistrue=latitude_axis_true,
                       log10eaxis=log10_eaxis, longitudeaxis=longitude_axis, latitudeaxis=latitude_axis,
                       logjacob=makelogjacob(log10_eaxis), save_directory=run_data_folder, 
                       logpsf=psf_test, logedisp=edisp_test, aeff=log_aeff)

    with open(f"{workingfolder}/{stemdirname}/singlerundata/inputconfig.yaml", 'w') as file:
            yaml.dump(stem_config_dict, file, default_flow_style=False)

    




    for runnum in range(1,numberofruns+1):
        single_run_data_folder = f"{workingfolder}/{stemdirname}/singlerundata/{runnum}"
        
        os.makedirs(single_run_data_folder)
        
        time.sleep(0.1)
        config_dict = stem_config_dict
        config_dict['runnumber'] = runnum
        
        with open(f"{single_run_data_folder}/inputconfig.yaml", 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)
        
        str =f"""#!/bin/bash
#
#SBATCH --job-name=SR{logmass}|{xi_true}|{runnum}|{int(math.log10(numberofruns*singlerunevents))}|{identifier}
#SBATCH --output=data/LatestFolder/SR{logmass}_{xi_true}_{runnum}_{int(numberofruns*singlerunevents)}_{identifier}.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={numcores}
#SBATCH --time={num_marg_hours}:{num_marg_minutes}:00
#SBATCH --mem-per-cpu={marginalisation_mem}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
conda init bash
conda activate DMPipe
srun python3 sim_and_marg_nuisance.py {single_run_data_folder}/inputconfig.yaml"""
        with open(f"{single_run_data_folder}/jobscript.sh", 'w') as f:
            f.write(str)
        if immediate_run:
            os.system(f"sbatch {single_run_data_folder}/jobscript.sh")

    strboi =f"""#!/bin/bash
#
#SBATCH --job-name=CR{logmass}|{xi_true}|{int(math.log10(numberofruns*singlerunevents))}|{identifier}
#SBATCH --output=data/LatestFolder/CR{logmass}_{xi_true}_{int(numberofruns*singlerunevents)}_{identifier}.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time={num_combine_hours}:{num_combine_minutes}:00
#SBATCH --mem-per-cpu={combination_mem}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
conda init bash
conda activate DMPipe
srun python3 combine_results.py {workingfolder}/{stemdirname}/singlerundata/inputconfig.yaml"""

    with open(f"{workingfolder}/{stemdirname}/CR.sh", 'w') as f:
        f.write(strboi)


# Following is so the module can be run as a primary script
if __name__=="__main__":
    logmass = float(sys.argv[1])
    xi_true = float(sys.argv[2])  
    numberofruns = int(sys.argv[3])
    singlerunevents = int(sys.argv[4])
    numcores = int(sys.argv[5])
    num_marg_hours = int(sys.argv[6])
    num_marg_minutes = int(sys.argv[7])
    num_combine_hours = int(sys.argv[8])
    num_combine_minutes = int(sys.argv[9])
    identifier = sys.argv[10]
    try:
        nbins_logmass = int(sys.argv[11])
    except:
        nbins_logmass = 101
        
    try:
        nbins_xi = int(sys.argv[12])
    except:
        nbins_xi = 161
        
    
    try:
        marginalisation_mem = int(sys.argv[13])
    except:
        marginalisation_mem = 200
        
    try:
        combination_mem = int(sys.argv[14])
    except:
        combination_mem = 1000
    try:
        dmdensity_profile = sys.argv[15]
    except:
        dmdensity_profile = 'einasto'
        

    try:
        num_true_energy_bins_per_decade = int(sys.argv[16])
    except:
        num_true_energy_bins_per_decade = 300

    try:
        num_recon_energy_bins_per_decade = int(sys.argv[17])
    except:
        num_recon_energy_bins_per_decade = 50

    try:
        true_spatial_res = float(sys.argv[18])
    except:
        true_spatial_res = 0.2
    try:
        recon_spatial_res = float(sys.argv[19])
    except:
        recon_spatial_res = 0.4


    try:
        immediate_run = int(sys.argv[20])
    except:
        immediate_run = 1

        

    makejobscripts(logmass=logmass, xi_true=xi_true, numberofruns=numberofruns, singlerunevents=singlerunevents, numcores=numcores, 
                   num_marg_hours=num_marg_hours, num_marg_minutes=num_marg_minutes, num_combine_hours=num_combine_hours, num_combine_minutes=num_combine_minutes, 
                   nbins_logmass=nbins_logmass, nbins_xi=nbins_xi, 
                   identifier=identifier, dmdensity_profile=dmdensity_profile,
                   marginalisation_mem=marginalisation_mem, combination_mem=combination_mem, 
                   immediate_run=immediate_run, 
                   num_true_energy_bins_per_decade=num_true_energy_bins_per_decade,
                   num_recon_energy_bins_per_decade=num_recon_energy_bins_per_decade,
                   true_spatial_res=true_spatial_res, recon_spatial_res=recon_spatial_res)
