#!/bin/bash
#
#SBATCH --job-name=CR2.0|0.8|3
#SBATCH --output=data/LatestFolder/CR2.0_0.8_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 recycling.py moretoy14 32