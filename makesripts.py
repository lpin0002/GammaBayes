

import os 
import sys
import numpy as np

lambdaval = float(sys.argv[1])
Nsamples = float(sys.argv[2])
logmass = float(sys.argv[3])
timestring = str(sys.argv[4])
timehour = int(sys.argv[5])
timeminute = int(sys.argv[6])
mem = int(sys.argv[7])

if logmass<0:
    strlogmass = f"n{np.abs(logmass)}"
else:
    strlogmass = f"{logmass}"

if timeminute<10:
    timeminute = f"0{timeminute}"

os.system(f"python simulation.py {lambdaval} {Nsamples} {logmass} {timestring}")

str = f"""#!/bin/bash                                                                                                                                                                                                                                        
#                                                                                                                                                                                                                                                  
#SBATCH --job-name=DM{strlogmass}_P_{timestring}                                                                                                                                                                                                                  
#SBATCH --output=runs/Outputs/DM{strlogmass}_{timestring}.txt                                                                                                                                                                                                   
#                                                                                                                                                                                                                                                  
#SBATCH --ntasks=1                                                                                                                                                                                                                                 
#SBATCH --cpus-per-task=1                                                                                                                                                                                                                          
#SBATCH --time={timehour}:{timeminute}:00                                                                                                                                                                                                                             
#SBATCH --mem-per-cpu={mem}                                                                                                                                                                                                                          
#SBATCH --mail-type=ALL                                                                                                                                                                                                                            
#SBATCH --mail-user=progressemail1999@gmail.com                                                                                                                                                                                                    
source activate DMPipe
srun python3 marginalisation.py {timestring}"""

workingfolder = os.path.realpath(os.path.join(sys.path[0]))
print(workingfolder)
os.system(f"mkdir {workingfolder}/runs")

os.system(f"mkdir {workingfolder}/runs/{timestring}")

with open(f"{workingfolder}/runs/{timestring}/SRM.sh", 'w') as f:
            f.write(str)

os.system(f"sbatch {workingfolder}/runs/{timestring}/SRM.sh")