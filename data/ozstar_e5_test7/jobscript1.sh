#!/bin/bash
#
#SBATCH --job-name=SR0.0|0.5|1|5
#SBATCH --output=data/LatestFolder/SR0.0_0.5_1_100000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6:30:00
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py ozstar_e5_test7 1 100000 0.0 0.5