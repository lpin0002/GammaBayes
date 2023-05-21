#!/bin/bash
#
#SBATCH --job-name=DM0.5|1.0|3|3
#SBATCH --output=data/LatestFolder/DM0.5_1.0_3_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=5:30:00
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py test4 3 200 0.5 1.0 10