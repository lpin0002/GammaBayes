#!/bin/bash
#
#SBATCH --job-name=DM2.0|1.0|3|3
#SBATCH --output=data/LatestFolder/DM2.0_1.0_3_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py moretoy7 3 100 2.0 1.0 20