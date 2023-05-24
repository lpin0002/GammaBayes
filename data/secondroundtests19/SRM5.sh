#!/bin/bash
#
#SBATCH --job-name=DM-0.25|0.2|5|3
#SBATCH --output=data/LatestFolder/DM-0.25_0.2_5_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=5:30:00
#SBATCH --mem-per-cpu=1600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py secondroundtests19 5 200 -0.25 0.2 10