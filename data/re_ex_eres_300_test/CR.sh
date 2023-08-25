#!/bin/bash
#
#SBATCH --job-name=CR0.0|0.0005|4|re_ex_eres_300_test
#SBATCH --output=data/LatestFolder/CR0.0_0.0005_10000_re_ex_eres_300_test.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=10000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 gridsearch.py re_ex_eres_300_test 161 1