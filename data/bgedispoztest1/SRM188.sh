#!/bin/bash
#
#SBATCH --job-name=DM1.0|0.8|188|4
#SBATCH --output=data/LatestFolder/DM1.0_0.8_188_10000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:10:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py bgedispoztest1 188 40 1.0 0.8 20