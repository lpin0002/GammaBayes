#!/bin/bash
#
#SBATCH --job-name=SR0.5|0.5|45|6|big_pp_test_2
#SBATCH --output=data/LatestFolder/SR0.5_0.5_45_1000000_big_pp_test_2.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:30:00
#SBATCH --mem-per-cpu=16500
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py big_pp_test_2 45 20000 0.5 0.5
srun python3 nuisancemarginalisation.py big_pp_test_2 45 50 61 8