#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.5|11|2
#SBATCH --output=data/LatestFolder/DM0.0_0.5_11_500.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py 1dgrid5 11 10 0.0 0.5 20