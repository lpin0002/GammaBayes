#!/bin/bash
#
#SBATCH --job-name=DM-0.8|0.5|1|3
#SBATCH --output=data/LatestFolder/DM-0.8_0.5_1_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:40:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py toylikespec9 1 100 -0.8 0.5 10