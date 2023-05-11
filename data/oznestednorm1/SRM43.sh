#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.5|43|2
#SBATCH --output=data/LatestFolder/DM0.0_0.5_43_500.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=500
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py oznestednorm1 43 10 0.0 0.5 10