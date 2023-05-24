#!/bin/bash
#
#SBATCH --job-name=DM-0.75|0.5|6|3
#SBATCH --output=data/LatestFolder/DM-0.75_0.5_6_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=5:30:00
#SBATCH --mem-per-cpu=1600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py secondroundtests12 6 200 -0.75 0.5 10