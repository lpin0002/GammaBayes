#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.8|3|2
#SBATCH --output=data/LatestFolder/DM0.0_0.8_3_500.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0:20:00
#SBATCH --mem-per-cpu=200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py ozenvtest4 3 20 0.0 0.8 20