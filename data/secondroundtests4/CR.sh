#!/bin/bash
#
#SBATCH --job-name=CR-0.25|1.0|3
#SBATCH --output=data/LatestFolder/CR-0.25_1.0_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=48:30:00
#SBATCH --mem-per-cpu=1600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 recycling.py secondroundtests4 32
