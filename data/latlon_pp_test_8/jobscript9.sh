#!/bin/bash
#
#SBATCH --job-name=SR0.0|0.5000000001|9|5
#SBATCH --output=data/LatestFolder/SR0.0_0.5000000001_9_100000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=3:10:00
#SBATCH --mem-per-cpu=1500
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py latlon_pp_test_8 9 10000 0.0 0.5000000001
srun python3 nuisancemarginalisation.py latlon_pp_test_8 9 10 101 10