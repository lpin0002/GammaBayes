#!/bin/bash
#
#SBATCH --job-name=DM0.5|0.5|2|3
#SBATCH --output=data/LatestFolder/DM0.5_0.5_2_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:40:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py tauchannel13 2 200 0.5 0.5 10