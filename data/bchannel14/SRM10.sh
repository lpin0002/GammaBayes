#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.5|10|3
#SBATCH --output=data/LatestFolder/DM0.0_0.5_10_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py bchannel14 10 50 0.0 0.5 20