#!/bin/bash
#
#SBATCH --job-name=CR-1.0|0.5|2
#SBATCH --output=data/LatestFolder/CR-1.0_0.5_100.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 recycling.py spectestgrid8 32