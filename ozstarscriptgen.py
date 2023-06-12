
import os, sys, numpy as np, time, math

def makejobscripts(logmass, ltrue, numberofruns, singlerunevents, numcores, numhour, numminute, identifier = None, immediate_run=1, nummemory = 200):
    
    if int(numminute)<10:
        numminute = "10"
    else:
        numminute = int(numminute)

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
#SBATCH --job-name=DM{logmass}|{ltrue}|{runnum}|{int(math.log10(numberofruns*singlerunevents))}
#SBATCH --output=data/LatestFolder/DM{logmass}_{ltrue}_{runnum}_{int(numberofruns*singlerunevents)}.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={numcores}
#SBATCH --time={numhour}:{numminute}:00
#SBATCH --mem-per-cpu={nummemory}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py {identifier} {runnum} {singlerunevents} {logmass} {ltrue} {numcores}
srun python3 calcposterior.py {identifier} {numcores}"""

        with open(f"{workingfolder}/{stemdirname}/jobscript{runnum}.sh", 'w') as f:
            f.write(str)
        if immediate_run:
            os.system(f"sbatch {workingfolder}/{stemdirname}/jobscript{runnum}.sh")

    

    with open(f"{workingfolder}/{stemdirname}/CR.sh", 'w') as f:
        f.write(str)

if __name__=="__main__":
    logmass = float(sys.argv[1])
    ltrue = float(sys.argv[2])  
    numberofruns = int(sys.argv[3])
    singlerunevents = int(sys.argv[4])
    numcores = int(sys.argv[5])
    numhour = int(sys.argv[6])
    numminute = int(sys.argv[7])
    identifier = sys.argv[8]
    try:
        nummemory = int(sys.argv[12])
    except:
        nummemory = 1000
    # def makejobscripts(logmass, ltrue, numberofruns, singlerunevents, numcores, numhour, numminute, identifier = None, immediate_run=1, nummemory = 200):

    makejobscripts(logmass=logmass, ltrue=ltrue, numberofruns=numberofruns, singlerunevents=singlerunevents, 
                numcores=numcores, numhour=numhour, numminute=numminute,
                identifier = identifier, nummemory = nummemory)