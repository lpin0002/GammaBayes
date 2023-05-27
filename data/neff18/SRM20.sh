#!/bin/bash
#
#SBATCH --job-name=DM0.25|0.2|20|3
#SBATCH --output=data/LatestFolder/DM0.25_0.2_20_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=1200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py neff18 20 50 0.25 0.2 20