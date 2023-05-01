#!/bin/bash
#
#SBATCH --job-name=DM0.5|0.5|197|4
#SBATCH --output=data/LatestFolder/DM0.5_0.5_197_10000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py fineoztest5 197 40 0.5 0.5 20