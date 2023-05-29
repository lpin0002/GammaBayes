#!/bin/bash
#
#SBATCH --job-name=DM-0.75|0.5|13|3
#SBATCH --output=data/LatestFolder/DM-0.75_0.5_13_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py bchannel11 13 50 -0.75 0.5 20