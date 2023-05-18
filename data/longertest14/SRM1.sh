#!/bin/bash
#
#SBATCH --job-name=DM2.0|0.8|1|3
#SBATCH --output=data/LatestFolder/DM2.0_0.8_1_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:50:00
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py longertest14 1 200 2.0 0.8 10