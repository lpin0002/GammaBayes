#!/bin/bash
#
#SBATCH --job-name=CR0.0|0.02|6|ozs_newbkg_3_2
#SBATCH --output=data/LatestFolder/CR0.0_0.02_1000000_ozs_newbkg_3_2.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=10000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 gridsearch.py ozs_newbkg_3_2 161 1