#!/bin/bash
#
#SBATCH --job-name=SR-0.8|0.5|5|4
#SBATCH --output=data/LatestFolder/SR-0.8_0.5_5_10000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=250
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py twodtestgrid4 5 400 -0.8 0.5