#!/bin/bash
#
#SBATCH --job-name=DM1.0|0.8|2|3
#SBATCH --output=data/LatestFolder/DM1.0_0.8_2_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=5:30:00
#SBATCH --mem-per-cpu=1600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py test12 2 200 1.0 0.8 10