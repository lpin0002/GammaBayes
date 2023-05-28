#!/bin/bash
#
#SBATCH --job-name=DM-0.5|0.8|4|3
#SBATCH --output=data/LatestFolder/DM-0.5_0.8_4_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py secondneff2 4 50 -0.5 0.8 20