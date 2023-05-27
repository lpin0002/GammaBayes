#!/bin/bash
#
#SBATCH --job-name=DM-0.25|0.2|32|3
#SBATCH --output=data/LatestFolder/DM-0.25_0.2_32_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py neff16 32 50 -0.25 0.2 20