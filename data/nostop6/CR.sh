#!/bin/bash
#
#SBATCH --job-name=CR1.5|1.0|2
#SBATCH --output=data/LatestFolder/CR1.5_1.0_500.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:30:00
#SBATCH --mem-per-cpu=1600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 recycling.py nostop6 32