#!/bin/bash
#
#SBATCH --job-name=SR0.5|0.5|3|4|latlon_newsetup_pp_test_3
#SBATCH --output=data/LatestFolder/SR0.5_0.5_3_10000_latlon_newsetup_pp_test_3.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=25000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py latlon_newsetup_pp_test_3 3 1000 0.5 0.5
srun python3 nuisancemarginalisation.py latlon_newsetup_pp_test_3 3 10 101 4