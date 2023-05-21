#!/bin/bash
#
#SBATCH --job-name=DM-0.5|1.0|9|3
#SBATCH --output=data/LatestFolder/DM-0.5_1.0_9_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=5:30:00
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py test2 9 200 -0.5 1.0 10