#!/bin/bash
#
#SBATCH --job-name=CR0.0|0.5|4|re_ex_eres_200_test4
#SBATCH --output=data/LatestFolder/CR0.0_0.5_20000_re_ex_eres_200_test4.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=10000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 gridsearch.py re_ex_eres_200_test4 161 1