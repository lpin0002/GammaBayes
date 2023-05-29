#!/bin/bash
#
#SBATCH --job-name=DM-0.25|0.2|18|3
#SBATCH --output=data/LatestFolder/DM-0.25_0.2_18_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py bchannel18 18 50 -0.25 0.2 20