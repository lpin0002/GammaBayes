#!/bin/bash
#
#SBATCH --job-name=CR0.5|0.5|4
#SBATCH --output=data/LatestFolder/CR0.5_0.5_10000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=144:30:00
#SBATCH --mem-per-cpu=5000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 recycling.py fineoztest5 32