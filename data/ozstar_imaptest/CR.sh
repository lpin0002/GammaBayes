#!/bin/bash
#
#SBATCH --job-name=CR0.5|0.5|3|ozstar_imaptest
#SBATCH --output=data/LatestFolder/CR0.5_0.5_1000_ozstar_imaptest.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 gridsearch.py ozstar_imaptest 161 1