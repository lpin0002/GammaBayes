#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.5|8|3
#SBATCH --output=data/LatestFolder/DM0.0_0.5_8_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:40:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py toylikespec13 8 100 0.0 0.5 10