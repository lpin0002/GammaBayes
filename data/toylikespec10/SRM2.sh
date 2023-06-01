#!/bin/bash
#
#SBATCH --job-name=DM-0.6|0.5|2|3
#SBATCH --output=data/LatestFolder/DM-0.6_0.5_2_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:40:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py toylikespec10 2 100 -0.6 0.5 10