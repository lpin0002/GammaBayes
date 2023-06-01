#!/bin/bash
#
#SBATCH --job-name=DM0.5|0.8|1|3
#SBATCH --output=data/LatestFolder/DM0.5_0.8_1_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:40:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py tauchannel6 1 200 0.5 0.8 10