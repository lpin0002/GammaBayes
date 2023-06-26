#!/bin/bash
#
#SBATCH --job-name=SR1.2|0.5|9|4
#SBATCH --output=data/LatestFolder/SR1.2_0.5_9_10000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=250
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py twodtestgrid6 9 400 1.2 0.5