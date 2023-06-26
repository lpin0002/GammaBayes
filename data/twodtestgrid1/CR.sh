#!/bin/bash
#
#SBATCH --job-name=CR-0.8|0.8|4
#SBATCH --output=data/LatestFolder/CR-0.8_0.8_10000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --time=24:30:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 calculateirfvalues.py twodtestgrid1 30
srun python3 gridsearch.py twodtestgrid1 161 161 30