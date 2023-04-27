#!/bin/bash
#
#SBATCH --job-name=CR1.0_0.8_3
#SBATCH --output=data/LatestFolder/CR1.0_0.8_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 recycling.py recoztest4 32
