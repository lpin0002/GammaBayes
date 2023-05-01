#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.4|148|3
#SBATCH --output=data/LatestFolder/DM0.0_0.4_148_5000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py fineoztest4 148 20 0.0 0.4 20