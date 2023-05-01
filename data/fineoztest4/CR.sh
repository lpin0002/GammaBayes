#!/bin/bash
#
#SBATCH --job-name=CR0.0|0.4|3
#SBATCH --output=data/LatestFolder/CR0.0_0.4_5000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:30:00
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 recycling.py fineoztest4 32