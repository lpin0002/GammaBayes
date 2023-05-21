#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.2|2|3
#SBATCH --output=data/LatestFolder/DM0.0_0.2_2_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=5:30:00
#SBATCH --mem-per-cpu=1600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py test24 2 200 0.0 0.2 10