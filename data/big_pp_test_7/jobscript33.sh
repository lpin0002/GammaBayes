#!/bin/bash
#
#SBATCH --job-name=SR0.5|0.5|33|6|big_pp_test_7
#SBATCH --output=data/LatestFolder/SR0.5_0.5_33_1000000_big_pp_test_7.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:30:00
#SBATCH --mem-per-cpu=16500
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py big_pp_test_7 33 10000 0.5 0.5
srun python3 nuisancemarginalisation.py big_pp_test_7 33 100 61 8