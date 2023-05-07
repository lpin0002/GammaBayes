#!/bin/bash
#
#SBATCH --job-name=DM1.0|0.8|202|4
#SBATCH --output=data/LatestFolder/DM1.0_0.8_202_10000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:10:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py bgedispoztest1 202 40 1.0 0.8 20