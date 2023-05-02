#!/bin/bash
#
#SBATCH --job-name=DM1.0|0.2|1|2
#SBATCH --output=data/LatestFolder/DM1.0_0.2_1_500.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py 1dgrid3 1 10 1.0 0.2 20