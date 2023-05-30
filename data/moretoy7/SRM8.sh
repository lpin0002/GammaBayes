#!/bin/bash
#
#SBATCH --job-name=DM2.0|1.0|8|3
#SBATCH --output=data/LatestFolder/DM2.0_1.0_8_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py moretoy7 8 100 2.0 1.0 20