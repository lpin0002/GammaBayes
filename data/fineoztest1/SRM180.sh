#!/bin/bash
#
#SBATCH --job-name=DM1.0|0.5|180|3
#SBATCH --output=data/LatestFolder/DM1.0_0.5_180_5000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py fineoztest1 180 20 1.0 0.5 20