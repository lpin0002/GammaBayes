#!/bin/bash
#
#SBATCH --job-name=CR0.0|0.5|4
#SBATCH --output=data/LatestFolder/CR0.0_0.5_10000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=50000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 calculateirfvalues.py ozstarbigcheck4 4
srun python3 gridsearch.py ozstarbigcheck4 51 51 4
