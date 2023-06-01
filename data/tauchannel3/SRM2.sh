#!/bin/bash
#
#SBATCH --job-name=DM-0.4|0.8|2|3
#SBATCH --output=data/LatestFolder/DM-0.4_0.8_2_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:40:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py tauchannel3 2 200 -0.4 0.8 10