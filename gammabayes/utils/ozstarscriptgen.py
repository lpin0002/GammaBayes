
import os, sys, numpy as np, time, math, yaml

def makejobscripts(logmass, xi_true, numberofruns, singlerunevents, numcores, 
                   numsimhour, numsimminute, numanalysehour, numanalyseminute, 
                   nbins_logmass, nbins_xi, identifier = None, immediate_run=1, 
                   simmemory = 200, analysememory=1000, dmdensity_profile='einasto'):
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
    if int(numsimminute)<10:
        numsimminute = "10"
    else:
        numsimminute = int(numsimminute)
        
    if int(numanalyseminute)<10:
        numanalyseminute = "10"
    else:
        numanalyseminute = int(numanalyseminute)

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
    
    os.makedirs(f"{workingfolder}/{stemdirname}/singlerundata", exist_ok=True)

    for runnum in range(1,numberofruns+1):
        single_run_data_folder = f"{workingfolder}/{stemdirname}/singlerundata/{runnum}"
        
        os.makedirs(single_run_data_folder)
        
        time.sleep(0.1)
        config_dict = {
            'identifier'        : identifier,
            'Nevents'           : singlerunevents,
            'logmass'           : logmass,
            'xi'                : xi_true,
            'nbins_logmass'     : nbins_logmass,
            'nbins_xi'          : nbins_xi,
            'dmdensity_profile' : dmdensity_profile,
            'numcores'          : numcores,
            'runnumber'         : runnum,
            'totalevents'       : singlerunevents*numberofruns,
                    }
        
        with open(f"{single_run_data_folder}/inputconfig.yaml", 'w') as file:
            yaml.dump(config_dict, file, default_flow_style=False)
        
        str =f"""#!/bin/bash
#
#SBATCH --job-name=SR{logmass}|{xi_true}|{runnum}|{int(math.log10(numberofruns*singlerunevents))}|{identifier}
#SBATCH --output=data/LatestFolder/SR{logmass}_{xi_true}_{runnum}_{int(numberofruns*singlerunevents)}_{identifier}.log
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={numcores}
#SBATCH --time={numsimhour}:{numsimminute}:00
#SBATCH --mem-per-cpu={simmemory}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source ~/.bashrc
srun python3 single_script_code.py {single_run_data_folder}/inputconfig.yaml"""
        with open(f"{single_run_data_folder}/jobscript.sh", 'w') as f:
            f.write(str)
        if immediate_run:
            os.system(f"sbatch {single_run_data_folder}/jobscript.sh")

    str =f"""#!/bin/bash
#
#SBATCH --job-name=CR{logmass}|{xi_true}|{int(math.log10(numberofruns*singlerunevents))}|{identifier}
#SBATCH --output=data/LatestFolder/CR{logmass}_{xi_true}_{int(numberofruns*singlerunevents)}_{identifier}.log
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time={numanalysehour}:{numanalyseminute}:00
#SBATCH --mem-per-cpu={analysememory}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source ~/.bashrc
srun python3 combine_results.py {workingfolder}/{stemdirname}/singlerundata/inputconfig.yaml"""

    with open(f"{workingfolder}/{stemdirname}/CR.sh", 'w') as f:
        f.write(str)




#logmass, xi_true, numberofruns, singlerunevents, numcores, 
                #    numsimhour, numsimminute, numanalysehour, numanalyseminute, 
                #    numlogmass, numlambda, identifier = None, immediate_run=1, 
                #    simmemory = 200, analysememory=1000


if __name__=="__main__":
    logmass = float(sys.argv[1])
    xi_true = float(sys.argv[2])  
    numberofruns = int(sys.argv[3])
    singlerunevents = int(sys.argv[4])
    numcores = int(sys.argv[5])
    numsimhour = int(sys.argv[6])
    numsimminute = int(sys.argv[7])
    numanalysehour = int(sys.argv[8])
    numanalyseminute = int(sys.argv[9])
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
        simmemory = int(sys.argv[13])
    except:
        simmemory = 200
        
        
    try:
        analysememory = int(sys.argv[14])
    except:
        analysememory = 1000
    try:
        dmdensity_profile = sys.argv[15]
    except:
        dmdensity_profile = 'einasto'
        
        
    try:
        immediate_run = int(sys.argv[16])
    except:
        immediate_run = 1
        

    makejobscripts(logmass=logmass, xi_true=xi_true, numberofruns=numberofruns, singlerunevents=singlerunevents, numcores=numcores, 
                   numsimhour=numsimhour, numsimminute=numsimminute, numanalysehour=numanalysehour, numanalyseminute=numanalyseminute, 
                   nbins_logmass=nbins_logmass, nbins_xi=nbins_xi, 
                   identifier=identifier, dmdensity_profile=dmdensity_profile,
                   simmemory=simmemory, analysememory=analysememory, 
                   immediate_run=immediate_run)
