#!/bin/bash
#
#SBATCH --job-name=CR0.0|0.5|5
#SBATCH --output=data/LatestFolder/CR0.0_0.5_100000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:30:00
#SBATCH --mem-per-cpu=400000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 calculateirfvalues.py ozstar_e5_test2 1
srun python3 gridsearch.py ozstar_e5_test2 51 51 1