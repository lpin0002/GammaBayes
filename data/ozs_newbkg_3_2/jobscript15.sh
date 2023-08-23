#!/bin/bash
#
#SBATCH --job-name=SR0.0|0.02|15|6|ozs_newbkg_3_2
#SBATCH --output=data/LatestFolder/SR0.0_0.02_15_1000000_ozs_newbkg_3_2.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=1100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py ozs_newbkg_3_2 15 50000 0.0 0.02
srun python3 nuisancemarginalisation.py ozs_newbkg_3_2 15 20 51 16