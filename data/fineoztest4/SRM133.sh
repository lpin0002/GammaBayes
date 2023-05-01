#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.4|133|3
#SBATCH --output=data/LatestFolder/DM0.0_0.4_133_5000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py fineoztest4 133 20 0.0 0.4 20