#!/bin/bash
#
#SBATCH --job-name=DM-1.3_49_0.5_3
#SBATCH --output=data/LatestFolder/DM-1.3_49_0.5_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:20:00
#SBATCH --mem-per-cpu=200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py ozstartest2 49 20 -1.3 0.5 10 41 161