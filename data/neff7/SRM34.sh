#!/bin/bash
#
#SBATCH --job-name=DM-1.0|0.5|34|3
#SBATCH --output=data/LatestFolder/DM-1.0_0.5_34_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py neff7 34 50 -1.0 0.5 20