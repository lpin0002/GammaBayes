#!/bin/bash
#
#SBATCH --job-name=DM-0.5|0.8|7|3
#SBATCH --output=data/LatestFolder/DM-0.5_0.8_7_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py bchannel22 7 50 -0.5 0.8 20