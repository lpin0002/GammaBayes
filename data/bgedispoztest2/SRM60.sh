#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.5|60|4
#SBATCH --output=data/LatestFolder/DM0.0_0.5_60_10000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:10:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py bgedispoztest2 60 40 0.0 0.5 20