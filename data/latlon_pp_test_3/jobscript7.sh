#!/bin/bash
#
#SBATCH --job-name=SR0.0|0.4999999|7|5
#SBATCH --output=data/LatestFolder/SR0.0_0.4999999_7_100000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=3:10:00
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py latlon_pp_test_3 7 10000 0.0 0.4999999
srun python3 nuisancemarginalisation.py latlon_pp_test_3 7 10 101 10