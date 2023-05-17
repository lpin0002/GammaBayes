#!/bin/bash
#
#SBATCH --job-name=DM1.5|0.2|2|2
#SBATCH --output=data/LatestFolder/DM1.5_0.2_2_500.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:50:00
#SBATCH --mem-per-cpu=900
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py nostop27 2 100 1.5 0.2 10