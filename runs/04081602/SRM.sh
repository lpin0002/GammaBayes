#!/bin/bash                                                                                                                                                                                                                                        
#                                                                                                                                                                                                                                                  
#SBATCH --job-name=MDM0.0_DMPipe_04081602                                                                                                                                                                                                                  
#SBATCH --output=runs/Outputs/MDM0.0_04081602.txt                                                                                                                                                                                                   
#                                                                                                                                                                                                                                                  
#SBATCH --ntasks=1                                                                                                                                                                                                                                 
#SBATCH --cpus-per-task=1                                                                                                                                                                                                                          
#SBATCH --time=1:20:00                                                                                                                                                                                                                             
#SBATCH --mem-per-cpu=600                                                                                                                                                                                                                          
#SBATCH --mail-type=ALL                                                                                                                                                                                                                            
#SBATCH --mail-user=progressemail1999@gmail.com                                                                                                                                                                                                    
source activate DMPipe
srun python3 marginalisation.py 04081602