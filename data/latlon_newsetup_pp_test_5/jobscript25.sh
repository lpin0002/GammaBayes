#!/bin/bash
#
#SBATCH --job-name=SR0.5|0.5|25|5|latlon_newsetup_pp_test_5
#SBATCH --output=data/LatestFolder/SR0.5_0.5_25_100000_latlon_newsetup_pp_test_5.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:30:00
#SBATCH --mem-per-cpu=17000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py latlon_newsetup_pp_test_5 25 2000 0.5 0.5
srun python3 nuisancemarginalisation.py latlon_newsetup_pp_test_5 25 50 61 6