#!/bin/bash
#
#SBATCH --job-name=DM-1.0|0.2|27|2
#SBATCH --output=data/LatestFolder/DM-1.0_0.2_27_500.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py 1dgrid9 27 10 -1.0 0.2 20