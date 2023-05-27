#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.2|23|3
#SBATCH --output=data/LatestFolder/DM0.0_0.2_23_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py neff17 23 50 0.0 0.2 20