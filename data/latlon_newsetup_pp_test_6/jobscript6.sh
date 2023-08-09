#!/bin/bash
#
#SBATCH --job-name=SR0.5|0.5|6|5|latlon_newsetup_pp_test_6
#SBATCH --output=data/LatestFolder/SR0.5_0.5_6_100000_latlon_newsetup_pp_test_6.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:30:00
#SBATCH --mem-per-cpu=16000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py latlon_newsetup_pp_test_6 6 5000 0.5 0.5
srun python3 nuisancemarginalisation.py latlon_newsetup_pp_test_6 6 20 61 8