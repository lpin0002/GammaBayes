#!/bin/bash
#
#SBATCH --job-name=DM1.0_78_0.8_3
#SBATCH --output=data/LatestFolder/DM1.0_78_0.8_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=400
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py recoztest4 78 10 1.0 0.8 10