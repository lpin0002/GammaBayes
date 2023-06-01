#!/bin/bash
#
#SBATCH --job-name=DM-1.0|0.5|10|3
#SBATCH --output=data/LatestFolder/DM-1.0_0.5_10_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:40:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py toylikespec8 10 100 -1.0 0.5 10