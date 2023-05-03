#!/bin/bash
#
#SBATCH --job-name=DM1.0|0.5|2|2
#SBATCH --output=data/LatestFolder/DM1.0_0.5_2_100.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py acctestgrid2 2 10 1.0 0.5 20