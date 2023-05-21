#!/bin/bash
#
#SBATCH --job-name=CR-1.0|0.2|3
#SBATCH --output=data/LatestFolder/CR-1.0_0.2_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:30:00
#SBATCH --mem-per-cpu=1600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 recycling.py test22 32