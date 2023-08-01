#!/bin/bash
#
#SBATCH --job-name=SR0.0|0.5|8|5
#SBATCH --output=data/LatestFolder/SR0.0_0.5_8_100000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=3:10:00
#SBATCH --mem-per-cpu=6000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py latlon_pp_test_1 8 10000 0.0 0.5
srun python3 nuisancemarginalisation.py latlon_pp_test_1 8 10 101 10