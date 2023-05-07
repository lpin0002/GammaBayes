#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.5|135|4
#SBATCH --output=data/LatestFolder/DM0.0_0.5_135_10000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:10:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py bgedispoztest2 135 40 0.0 0.5 20