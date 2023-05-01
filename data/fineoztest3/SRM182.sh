#!/bin/bash
#
#SBATCH --job-name=DM-1.0|0.2|182|3
#SBATCH --output=data/LatestFolder/DM-1.0_0.2_182_5000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py fineoztest3 182 20 -1.0 0.2 20