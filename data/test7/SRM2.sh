#!/bin/bash
#
#SBATCH --job-name=DM2.0|1.0|2|3
#SBATCH --output=data/LatestFolder/DM2.0_1.0_2_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=5:30:00
#SBATCH --mem-per-cpu=1600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py test7 2 200 2.0 1.0 10