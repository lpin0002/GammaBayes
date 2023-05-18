#!/bin/bash
#
#SBATCH --job-name=CR0.5|0.0|3
#SBATCH --output=data/LatestFolder/CR0.5_0.0_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:30:00
#SBATCH --mem-per-cpu=3000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 recycling.py longertest29 32