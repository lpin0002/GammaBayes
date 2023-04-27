#!/bin/bash
#
#SBATCH --job-name=DM1.0_44_0.2_3
#SBATCH --output=data/LatestFolder/DM1.0_44_0.2_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=400
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py recoztest7 44 10 1.0 0.2 10