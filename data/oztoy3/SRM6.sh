#!/bin/bash
#
#SBATCH --job-name=DM0.0|1.0|6|3
#SBATCH --output=data/LatestFolder/DM0.0_1.0_6_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py oztoy3 6 100 0.0 1.0 20