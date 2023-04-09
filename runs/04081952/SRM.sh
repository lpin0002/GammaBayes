#!/bin/bash                                                                                                                                                                                                                                        
#                                                                                                                                                                                                                                                  
#SBATCH --job-name=MDM2.0_DMPipe_04081952                                                                                                                                                                                                                  
#SBATCH --output=runs/Outputs/MDM2.0_04081952.txt                                                                                                                                                                                                   
#                                                                                                                                                                                                                                                  
#SBATCH --ntasks=1                                                                                                                                                                                                                                 
#SBATCH --cpus-per-task=1                                                                                                                                                                                                                          
#SBATCH --time=1:30:00                                                                                                                                                                                                                             
#SBATCH --mem-per-cpu=4000                                                                                                                                                                                                                          
#SBATCH --mail-type=ALL                                                                                                                                                                                                                            
#SBATCH --mail-user=progressemail1999@gmail.com                                                                                                                                                                                                    
source activate DMPipe
srun python3 marginalisation.py 04081952