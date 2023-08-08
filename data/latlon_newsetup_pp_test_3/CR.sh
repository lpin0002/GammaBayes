#!/bin/bash
#
#SBATCH --job-name=CR0.5|0.5|4|latlon_newsetup_pp_test_3
#SBATCH --output=data/LatestFolder/CR0.5_0.5_10000_latlon_newsetup_pp_test_3.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 gridsearch.py latlon_newsetup_pp_test_3 161 1