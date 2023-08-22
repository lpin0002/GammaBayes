#!/bin/bash
#
#SBATCH --job-name=CR0.0|0.5|4|ozs_newbkg
#SBATCH --output=data/LatestFolder/CR0.0_0.5_20000_ozs_newbkg.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=10000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 gridsearch.py ozs_newbkg 161 1