#!/bin/bash
#
#SBATCH --job-name=CR0.2|0.8|4
#SBATCH --output=data/LatestFolder/CR0.2_0.8_10000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --time=24:30:00
#SBATCH --mem-per-cpu=640
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 calculateirfvalues.py twodtestgrid2 30
srun python3 gridsearch.py twodtestgrid2 161 161 30
