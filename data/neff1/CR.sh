#!/bin/bash
#
#SBATCH --job-name=CR-1.0|0.8|3
#SBATCH --output=data/LatestFolder/CR-1.0_0.8_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:30:00
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 recycling.py neff1 32