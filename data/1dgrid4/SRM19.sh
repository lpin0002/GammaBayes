#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.8|19|2
#SBATCH --output=data/LatestFolder/DM0.0_0.8_19_500.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py 1dgrid4 19 10 0.0 0.8 20