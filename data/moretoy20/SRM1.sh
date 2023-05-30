#!/bin/bash
#
#SBATCH --job-name=DM1.5|0.5|1|3
#SBATCH --output=data/LatestFolder/DM1.5_0.5_1_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py moretoy20 1 100 1.5 0.5 20