#!/bin/bash                                                                                                                                                                                                                                        
#                                                                                                                                                                                                                                                  
#SBATCH --job-name=MDMn1.3_DMPipe_04090904                                                                                                                                                                                                                  
#SBATCH --output=runs/Outputs/MDMn1.3_04090904.txt                                                                                                                                                                                                   
#                                                                                                                                                                                                                                                  
#SBATCH --ntasks=1                                                                                                                                                                                                                                 
#SBATCH --cpus-per-task=1                                                                                                                                                                                                                          
#SBATCH --time=2:30:00                                                                                                                                                                                                                             
#SBATCH --mem-per-cpu=6000                                                                                                                                                                                                                          
#SBATCH --mail-type=ALL                                                                                                                                                                                                                            
#SBATCH --mail-user=progressemail1999@gmail.com                                                                                                                                                                                                    
source activate DMPipe
srun python3 marginalisation.py 04090904