#!/bin/bash
#
#SBATCH --job-name=SR0.0|0.5|1|4|ozs_newbkg
#SBATCH --output=data/LatestFolder/SR0.0_0.5_1_20000_ozs_newbkg.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:10:00
#SBATCH --mem-per-cpu=2000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py ozs_newbkg 1 10000 0.0 0.5
srun python3 nuisancemarginalisation.py ozs_newbkg 1 2 51 16