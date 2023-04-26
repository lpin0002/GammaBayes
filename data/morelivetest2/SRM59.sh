#!/bin/bash
#
#SBATCH --job-name=DM-1.0_59_0.5_3
#SBATCH --output=data/LatestFolder/DM-1.0_59_0.5_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:10:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py morelivetest2 59 10 -1.0 0.5 10 21 161 1000