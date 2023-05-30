#!/bin/bash
#
#SBATCH --job-name=DM-1.0|0.5|2|3
#SBATCH --output=data/LatestFolder/DM-1.0_0.5_2_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py moretoy15 2 100 -1.0 0.5 20