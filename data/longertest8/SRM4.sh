#!/bin/bash
#
#SBATCH --job-name=DM-1.0|0.8|4|3
#SBATCH --output=data/LatestFolder/DM-1.0_0.8_4_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:50:00
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py longertest8 4 200 -1.0 0.8 10