#!/bin/bash
#
#SBATCH --job-name=DM-0.25|0.8|14|3
#SBATCH --output=data/LatestFolder/DM-0.25_0.8_14_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py bchannel8 14 50 -0.25 0.8 20