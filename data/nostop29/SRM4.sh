#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.0|4|2
#SBATCH --output=data/LatestFolder/DM0.0_0.0_4_500.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:50:00
#SBATCH --mem-per-cpu=900
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py nostop29 4 100 0.0 0.0 10