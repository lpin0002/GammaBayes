#!/bin/bash
#
#SBATCH --job-name=CR0.0|0.5|3
#SBATCH --output=data/LatestFolder/CR0.0_0.5_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=3000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 calculateirfvalues.py ozstarcheck1 32
srun python3 gridsearch.py ozstarcheck1 51 101 32