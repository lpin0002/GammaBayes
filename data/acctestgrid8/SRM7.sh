#!/bin/bash
#
#SBATCH --job-name=DM-1.0|0.5|7|2
#SBATCH --output=data/LatestFolder/DM-1.0_0.5_7_100.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py acctestgrid8 7 10 -1.0 0.5 20