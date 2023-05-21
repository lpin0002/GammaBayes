#!/bin/bash
#
#SBATCH --job-name=DM0.5|0.8|5|3
#SBATCH --output=data/LatestFolder/DM0.5_0.8_5_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=5:30:00
#SBATCH --mem-per-cpu=1600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py test11 5 200 0.5 0.8 10