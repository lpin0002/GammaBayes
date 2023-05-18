#!/bin/bash
#
#SBATCH --job-name=DM0.5|0.2|1|3
#SBATCH --output=data/LatestFolder/DM0.5_0.2_1_2000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=2:50:00
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py longertest25 1 200 0.5 0.2 10