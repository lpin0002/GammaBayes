#!/bin/bash
#
#SBATCH --job-name=DM-0.25|1.0|4|3
#SBATCH --output=data/LatestFolder/DM-0.25_1.0_4_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=5:30:00
#SBATCH --mem-per-cpu=1600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py secondroundtests4 4 200 -0.25 1.0 10