#!/bin/bash
#
#SBATCH --job-name=DM-0.5|0.5|8|3
#SBATCH --output=data/LatestFolder/DM-0.5_0.5_8_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=6:30:00
#SBATCH --mem-per-cpu=1600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py neffective2 8 50 -0.5 0.5 20