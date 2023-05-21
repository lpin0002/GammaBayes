#!/bin/bash
#
#SBATCH --job-name=DM2.0|0.2|7|3
#SBATCH --output=data/LatestFolder/DM2.0_0.2_7_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=5:30:00
#SBATCH --mem-per-cpu=1600
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py test28 7 200 2.0 0.2 10