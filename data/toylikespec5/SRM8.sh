#!/bin/bash
#
#SBATCH --job-name=DM-0.2|0.8|8|3
#SBATCH --output=data/LatestFolder/DM-0.2_0.8_8_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:40:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py toylikespec5 8 100 -0.2 0.8 10