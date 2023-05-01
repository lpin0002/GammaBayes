#!/bin/bash
#
#SBATCH --job-name=DM-1.0|0.2|21|3
#SBATCH --output=data/LatestFolder/DM-1.0_0.2_21_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py fineoztest3_take4_ozstar 21 10 -1.0 0.2 10