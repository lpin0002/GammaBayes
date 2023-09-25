
import os, sys, numpy as np, time, math

def makejobscripts(logmass, xi_true, numberofruns, singlerunevents, numcores, 
                   numsimhour, numsimminute, numanalysehour, numanalyseminute, 
                   numlogmass, numlambda, identifier = None, immediate_run=1, 
                   simmemory = 200, analysememory=1000, densityprofile='einasto'):
    
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

    for runnum in range(1,numberofruns+1):
        time.sleep(0.1)
        #TODO: Adjust time allocation based on number of cores, accuracy and number of events
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
source activate DMPipe
srun python3 single_script_code.py {singlerunevents} {xi_true} {logmass} {identifier} {numlogmass} {numcores} {densityprofile} {runnum} {int(numberofruns*singlerunevents)}"""
        with open(f"{workingfolder}/{stemdirname}/jobscript{runnum}.sh", 'w') as f:
            f.write(str)
        if immediate_run:
            os.system(f"sbatch {workingfolder}/{stemdirname}/jobscript{runnum}.sh")

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
source activate DMPipe
srun python3 combine_results.py {identifier} {numlambda}"""

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
        densityprofile = sys.argv[15]
    except:
        densityprofile = 'einasto'
        
        
    try:
        immediate_run = int(sys.argv[16])
    except:
        immediate_run = 1
        

    makejobscripts(logmass=logmass, xi_true=xi_true, numberofruns=numberofruns, singlerunevents=singlerunevents, numcores=numcores, 
                   numsimhour=numsimhour, numsimminute=numsimminute, numanalysehour=numanalysehour, numanalyseminute=numanalyseminute, 
                   numlogmass=numlogmass, numlambda=numlambda, 
                   identifier=identifier, densityprofile=densityprofile,
                   simmemory=simmemory, analysememory=analysememory, 
                   immediate_run=immediate_run)