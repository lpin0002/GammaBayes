#!/bin/bash
#
#SBATCH --job-name=DM1.0|0.5|5|3
#SBATCH --output=data/LatestFolder/DM1.0_0.5_5_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:40:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py tauchannel14 5 200 1.0 0.5 10