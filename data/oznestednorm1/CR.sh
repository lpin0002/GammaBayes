#!/bin/bash
#
#SBATCH --job-name=CR0.0|0.5|2
#SBATCH --output=data/LatestFolder/CR0.0_0.5_500.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:10:00
#SBATCH --mem-per-cpu=500
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 recycling.py oznestednorm1 32