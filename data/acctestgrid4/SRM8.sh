#!/bin/bash
#
#SBATCH --job-name=DM0.0|0.8|8|2
#SBATCH --output=data/LatestFolder/DM0.0_0.8_8_100.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py acctestgrid4 8 10 0.0 0.8 20