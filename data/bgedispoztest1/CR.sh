#!/bin/bash
#
#SBATCH --job-name=CR1.0|0.8|4
#SBATCH --output=data/LatestFolder/CR1.0_0.8_10000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=144:10:00
#SBATCH --mem-per-cpu=5000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 recycling.py bgedispoztest1 32
