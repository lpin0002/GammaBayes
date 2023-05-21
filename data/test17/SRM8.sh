#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.5|8|3
#SBATCH --output=data/LatestFolder/DM0.0_0.5_8_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=5:30:00
#SBATCH --mem-per-cpu=1600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py test17 8 200 0.0 0.5 10