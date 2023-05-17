#!/bin/bash
#
#SBATCH --job-name=CR1.0|0.2|2
#SBATCH --output=data/LatestFolder/CR1.0_0.2_500.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:30:00
#SBATCH --mem-per-cpu=1600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 recycling.py nostop26 32