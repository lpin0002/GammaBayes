#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.8|4|3
#SBATCH --output=data/LatestFolder/DM0.0_0.8_4_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:50:00
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py longertest10 4 200 0.0 0.8 10