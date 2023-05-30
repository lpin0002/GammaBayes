#!/bin/bash
#
#SBATCH --job-name=DM2.0|0.8|5|3
#SBATCH --output=data/LatestFolder/DM2.0_0.8_5_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py moretoy14 5 100 2.0 0.8 20