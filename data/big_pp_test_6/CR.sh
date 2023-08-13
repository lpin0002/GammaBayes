#!/bin/bash
#
#SBATCH --job-name=CR0.5|0.5|6|big_pp_test_6
#SBATCH --output=data/LatestFolder/CR0.5_0.5_1000000_big_pp_test_6.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 gridsearch.py big_pp_test_6 101 1