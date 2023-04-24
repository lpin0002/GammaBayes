
import os, sys, numpy as np, time, math

def makejobscripts(logmass, ltrue, numberofruns, singlerunevents, margcores, marghour, margminute, identifier = None, nbinslogmass=21, nbinslambda=81, immediate_run=1, margmemory = 200):
    
    if int(margminute)<10:
        margminute = "10"
    else:
        margminute = int(margminute)

    workingfolder = os.path.realpath(os.path.join(sys.path[0]))

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

    for runnum in range(1,numberofruns+1):
        #TODO: Adjust time allocation based on number of cores, accuracy and number of events
        str =f"""#!/bin/bash
#
#SBATCH --job-name=DM{logmass}_{runnum}_{ltrue}_{int(math.log10(numberofruns*singlerunevents))}
#SBATCH --output=data/LatestFolder/DM{logmass}_{runnum}_{ltrue}_{int(numberofruns*singlerunevents)}.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={margcores}
#SBATCH --time={marghour}:{margminute}:00
#SBATCH --mem-per-cpu={margmemory}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py {identifier} {runnum} {singlerunevents} {logmass} {ltrue} {nbinslogmass} {nbinslambda}"""

        with open(f"{workingfolder}/{stemdirname}/SRM{runnum}.sh", 'w') as f:
            f.write(str)
        if immediate_run:
            os.system(f"sbatch {workingfolder}/{stemdirname}/SRM{runnum}.sh")

logmass = float(sys.argv[1])
ltrue = float(sys.argv[2])
numberofruns = int(sys.argv[3])
singlerunevents = int(sys.argv[4])
margcores = int(sys.argv[5])
marghour = int(sys.argv[6])
margminute = int(sys.argv[7])
identifier = sys.argv[8]

try:
    nbinslogmass = int(sys.argv[9])
except:
    nbinslogmass = 21

try:
    nbinslambda=int(sys.argv[10])
except:
    nbinslambda = 81

try:
    margmemory = int(sys.argv[11])
except:
    margmemory = 200

makejobscripts(logmass=logmass, ltrue=ltrue, numberofruns=numberofruns, singlerunevents=singlerunevents, 
               margcores=margcores, marghour=marghour, margminute=margminute, identifier = identifier, nbinslogmass=nbinslogmass, nbinslambda=nbinslambda, margmemory = 200)