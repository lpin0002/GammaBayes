#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.5|3|3
#SBATCH --output=data/LatestFolder/DM0.0_0.5_3_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:40:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py tauchannel12 3 200 0.0 0.5 10