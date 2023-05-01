#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.8|13|3
#SBATCH --output=data/LatestFolder/DM0.0_0.8_13_5000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py fineoztest2 13 20 0.0 0.8 20