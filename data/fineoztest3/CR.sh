#!/bin/bash
#
#SBATCH --job-name=CR-1.0|0.2|3
#SBATCH --output=data/LatestFolder/CR-1.0_0.2_5000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=72:30:00
#SBATCH --mem-per-cpu=3000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 recycling.py fineoztest3 32