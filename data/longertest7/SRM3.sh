#!/bin/bash
#
#SBATCH --job-name=DM2.0|1.0|3|3
#SBATCH --output=data/LatestFolder/DM2.0_1.0_3_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:50:00
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py longertest7 3 200 2.0 1.0 10