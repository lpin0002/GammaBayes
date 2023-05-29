#!/bin/bash
#
#SBATCH --job-name=CR1.0|0.5|3
#SBATCH --output=data/LatestFolder/CR1.0_0.5_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=1800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 recycling.py oztoy15 32