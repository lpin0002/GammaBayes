#!/bin/bash
#
#SBATCH --job-name=DM-0.5|0.2|29|3
#SBATCH --output=data/LatestFolder/DM-0.5_0.2_29_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py neff15 29 50 -0.5 0.2 20