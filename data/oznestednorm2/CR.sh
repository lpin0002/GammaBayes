#!/bin/bash
#
#SBATCH --job-name=CR0.1|0.5|2
#SBATCH --output=data/LatestFolder/CR0.1_0.5_500.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:10:00
#SBATCH --mem-per-cpu=400
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 recycling.py oznestednorm2 32