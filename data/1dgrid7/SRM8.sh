#!/bin/bash
#
#SBATCH --job-name=DM-1.0|0.8|8|2
#SBATCH --output=data/LatestFolder/DM-1.0_0.8_8_500.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py 1dgrid7 8 10 -1.0 0.8 20