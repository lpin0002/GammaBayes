#!/bin/bash
#
#SBATCH --job-name=DM0.5|0.5|77|4
#SBATCH --output=data/LatestFolder/DM0.5_0.5_77_10000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py fineoztest5 77 40 0.5 0.5 20