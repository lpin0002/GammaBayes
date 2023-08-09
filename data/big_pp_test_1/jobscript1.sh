#!/bin/bash
#
#SBATCH --job-name=SR0.5|0.5|1|6|big_pp_test_1
#SBATCH --output=data/LatestFolder/SR0.5_0.5_1_1000000_big_pp_test_1.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:30:00
#SBATCH --mem-per-cpu=16500
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py big_pp_test_1 1 20000 0.5 0.5
srun python3 nuisancemarginalisation.py big_pp_test_1 1 50 61 8