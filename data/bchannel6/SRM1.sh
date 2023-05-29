#!/bin/bash
#
#SBATCH --job-name=DM-0.75|1.0|1|3
#SBATCH --output=data/LatestFolder/DM-0.75_1.0_1_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py bchannel6 1 50 -0.75 1.0 20