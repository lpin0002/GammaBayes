#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.5|29|3
#SBATCH --output=data/LatestFolder/DM0.0_0.5_29_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:10:00
#SBATCH --mem-per-cpu=500
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py newspec_oz1 29 20 0.0 0.5 20