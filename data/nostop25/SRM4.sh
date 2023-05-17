#!/bin/bash
#
#SBATCH --job-name=DM0.5|0.2|4|2
#SBATCH --output=data/LatestFolder/DM0.5_0.2_4_500.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:50:00
#SBATCH --mem-per-cpu=900
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py nostop25 4 100 0.5 0.2 10