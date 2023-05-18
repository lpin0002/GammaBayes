#!/bin/bash
#
#SBATCH --job-name=DM1.5|0.2|2|3
#SBATCH --output=data/LatestFolder/DM1.5_0.2_2_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:50:00
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py longertest27 2 200 1.5 0.2 10