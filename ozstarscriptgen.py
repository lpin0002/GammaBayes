
import os, sys, numpy as np, time, math

def makejobscripts(logmass, ltrue, numberofruns, singlerunevents, numcores, 
                   numsimhour, numsimminute, numanalysehour, numanalyseminute, 
                   numlogmass, numlambda, identifier = None, immediate_run=1, 
                   simmemory = 200, analysememory=1000):
    
    if int(numsimminute)<10:
        numsimminute = "10"
    else:
        numsimminute = int(numsimminute)
        
    if int(numanalyseminute)<10:
        numanalyseminute = "10"
    else:
        numanalyseminute = int(numanalyseminute)

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
#SBATCH --job-name=SR{logmass}|{ltrue}|{runnum}|{int(math.log10(numberofruns*singlerunevents))}
#SBATCH --output=data/LatestFolder/SR{logmass}_{ltrue}_{runnum}_{int(numberofruns*singlerunevents)}.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={numcores}
#SBATCH --time={numsimhour}:{numsimminute}:00
#SBATCH --mem-per-cpu={simmemory}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py {identifier} {runnum} {singlerunevents} {logmass} {ltrue}
srun python3 calculateirfvalues.py {identifier} {runnum} {numberofruns} {numlogmass} {numlambda} {numcores}"""
        with open(f"{workingfolder}/{stemdirname}/jobscript{runnum}.sh", 'w') as f:
            f.write(str)
        if immediate_run:
            os.system(f"sbatch {workingfolder}/{stemdirname}/jobscript{runnum}.sh")

    str =f"""#!/bin/bash
#
#SBATCH --job-name=CR{logmass}|{ltrue}|{int(math.log10(numberofruns*singlerunevents))}
#SBATCH --output=data/LatestFolder/CR{logmass}_{ltrue}_{int(numberofruns*singlerunevents)}.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time={numanalysehour}:{numanalyseminute}:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 gridsearch.py {identifier}"""

    with open(f"{workingfolder}/{stemdirname}/CR.sh", 'w') as f:
        f.write(str)




#logmass, ltrue, numberofruns, singlerunevents, numcores, 
                #    numsimhour, numsimminute, numanalysehour, numanalyseminute, 
                #    numlogmass, numlambda, identifier = None, immediate_run=1, 
                #    simmemory = 200, analysememory=1000


if __name__=="__main__":
    logmass = float(sys.argv[1])
    ltrue = float(sys.argv[2])  
    numberofruns = int(sys.argv[3])
    singlerunevents = int(sys.argv[4])
    numcores = int(sys.argv[5])
    numsimhour = int(sys.argv[6])
    numsimminute = int(sys.argv[7])
    numanalysehour = int(sys.argv[8])
    numanalyseminute = int(sys.argv[9])
    identifier = sys.argv[10]
    try:
        numlogmass = int(sys.argv[11])
    except:
        numlogmass = 101
        
    try:
        numlambda = int(sys.argv[12])
    except:
        numlambda = 161
        
    
    try:
        simmemory = int(sys.argv[13])
    except:
        simmemory = 200
        
        
    try:
        analysememory = int(sys.argv[14])
    except:
        analysememory = 1000
        
    try:
        immediate_run = int(sys.argv[15])
    except:
        immediate_run = 1
        

    makejobscripts(logmass=logmass, ltrue=ltrue, numberofruns=numberofruns, singlerunevents=singlerunevents, numcores=numcores, 
                   numsimhour=numsimhour, numsimminute=numsimminute, numanalysehour=numanalysehour, numanalyseminute=numanalyseminute, 
                   numlogmass=numlogmass, numlambda=numlambda, 
                   identifier=identifier, 
                   simmemory=simmemory, analysememory=analysememory, 
                   immediate_run=immediate_run)