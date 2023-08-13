#!/bin/bash
#
#SBATCH --job-name=CR0.5|0.5|6|big_pp_test_5
#SBATCH --output=data/LatestFolder/CR0.5_0.5_1000000_big_pp_test_5.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=16000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 gridsearch.py big_pp_test_5 101 1
