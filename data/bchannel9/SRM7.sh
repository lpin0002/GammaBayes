#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.8|7|3
#SBATCH --output=data/LatestFolder/DM0.0_0.8_7_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py bchannel9 7 50 0.0 0.8 20