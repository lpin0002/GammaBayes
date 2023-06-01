#!/bin/bash
#
#SBATCH --job-name=DM0.2|0.8|6|3
#SBATCH --output=data/LatestFolder/DM0.2_0.8_6_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:40:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py toylikespec7 6 100 0.2 0.8 10