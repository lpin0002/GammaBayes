
import os, sys, numpy as np, time, math, yaml

def makejobscripts(logmass, xi_true, numberofruns, singlerunevents, numcores, 
                   numsimhour, numsimminute, numanalysehour, numanalyseminute, 
                   nbins_logmass, nbins_xi, identifier = None, immediate_run=1, 
                   simmemory = 200, analysememory=1000, dmdensity_profile='einasto'):
    
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
#SBATCH --output=data/LatestFolder/SR{logmass}_{xi_true}_{runnum}_{int(numberofruns*singlerunevents)}_{identifier}.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={numcores}
#SBATCH --time={numsimhour}:{numsimminute}:00
#SBATCH --mem-per-cpu={simmemory}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
conda init bash
conda activate DMPipe
srun python3 sim_and_nuisance_marg.py {single_run_data_folder}/inputconfig.yaml"""
        with open(f"{single_run_data_folder}/jobscript.sh", 'w') as f:
            f.write(str)
        if immediate_run:
            os.system(f"sbatch {single_run_data_folder}/jobscript.sh")

    str =f"""#!/bin/bash
#
#SBATCH --job-name=CR{logmass}|{xi_true}|{int(math.log10(numberofruns*singlerunevents))}|{identifier}
#SBATCH --output=data/LatestFolder/CR{logmass}_{xi_true}_{int(numberofruns*singlerunevents)}_{identifier}.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time={numanalysehour}:{numanalyseminute}:00
#SBATCH --mem-per-cpu={analysememory}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
conda init bash
conda activate DMPipe
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
