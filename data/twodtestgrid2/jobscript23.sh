#!/bin/bash
#
#SBATCH --job-name=SR0.2|0.8|23|4
#SBATCH --output=data/LatestFolder/SR0.2_0.8_23_10000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=250
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py twodtestgrid2 23 400 0.2 0.8