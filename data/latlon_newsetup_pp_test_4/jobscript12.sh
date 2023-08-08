#!/bin/bash
#
#SBATCH --job-name=SR0.5|0.5|12|5|latlon_newsetup_pp_test_4
#SBATCH --output=data/LatestFolder/SR0.5_0.5_12_100000_latlon_newsetup_pp_test_4.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --time=4:30:00
#SBATCH --mem-per-cpu=21000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py latlon_newsetup_pp_test_4 12 5000 0.5 0.5
srun python3 nuisancemarginalisation.py latlon_newsetup_pp_test_4 12 20 61 6