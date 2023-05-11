#!/bin/bash
#
#SBATCH --job-name=DM0.1|0.5|24|2
#SBATCH --output=data/LatestFolder/DM0.1_0.5_24_500.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=300
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py oznestednorm2 24 10 0.1 0.5 10