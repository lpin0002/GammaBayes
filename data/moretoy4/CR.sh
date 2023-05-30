#!/bin/bash
#
#SBATCH --job-name=CR0.5|1.0|3
#SBATCH --output=data/LatestFolder/CR0.5_1.0_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 recycling.py moretoy4 32
