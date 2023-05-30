#!/bin/bash
#
#SBATCH --job-name=DM1.0|0.8|7|3
#SBATCH --output=data/LatestFolder/DM1.0_0.8_7_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py moretoy12 7 100 1.0 0.8 20