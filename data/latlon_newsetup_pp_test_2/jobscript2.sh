#!/bin/bash
#
#SBATCH --job-name=SR0.5|0.5|2|5
#SBATCH --output=data/LatestFolder/SR0.5_0.5_2_100000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=6:10:00
#SBATCH --mem-per-cpu=14000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py latlon_newsetup_pp_test_2 2 10000 0.5 0.5
srun python3 nuisancemarginalisation.py latlon_newsetup_pp_test_2 2 10 101 10