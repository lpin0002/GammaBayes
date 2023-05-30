#!/bin/bash
#
#SBATCH --job-name=DM0.5|1.0|5|3
#SBATCH --output=data/LatestFolder/DM0.5_1.0_5_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py moretoy4 5 100 0.5 1.0 20