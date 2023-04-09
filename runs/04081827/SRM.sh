#!/bin/bash                                                                                                                                                                                                                                        
#                                                                                                                                                                                                                                                  
#SBATCH --job-name=MDM1.0_DMPipe_04081827                                                                                                                                                                                                                  
#SBATCH --output=runs/Outputs/MDM1.0_04081827.txt                                                                                                                                                                                                   
#                                                                                                                                                                                                                                                  
#SBATCH --ntasks=1                                                                                                                                                                                                                                 
#SBATCH --cpus-per-task=1                                                                                                                                                                                                                          
#SBATCH --time=1:00:00                                                                                                                                                                                                                             
#SBATCH --mem-per-cpu=5000                                                                                                                                                                                                                          
#SBATCH --mail-type=ALL                                                                                                                                                                                                                            
#SBATCH --mail-user=progressemail1999@gmail.com                                                                                                                                                                                                    
source activate DMPipe
srun python3 marginalisation.py 04081827