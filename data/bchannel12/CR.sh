#!/bin/bash
#
#SBATCH --job-name=CR-0.5|0.5|3
#SBATCH --output=data/LatestFolder/CR-0.5_0.5_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:30:00
#SBATCH --mem-per-cpu=1800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 recycling.py bchannel12 32