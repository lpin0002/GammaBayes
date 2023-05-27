#!/bin/bash
#
#SBATCH --job-name=DM-0.25|0.8|15|3
#SBATCH --output=data/LatestFolder/DM-0.25_0.8_15_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py neff4 15 50 -0.25 0.8 20