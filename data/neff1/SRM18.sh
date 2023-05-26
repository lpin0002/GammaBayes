#!/bin/bash
#
#SBATCH --job-name=DM-1.0|0.8|18|3
#SBATCH --output=data/LatestFolder/DM-1.0_0.8_18_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py neff1 18 50 -1.0 0.8 20