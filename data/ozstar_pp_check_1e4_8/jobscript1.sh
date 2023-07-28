#!/bin/bash
#
#SBATCH --job-name=SR0.0|0.5|1|4
#SBATCH --output=data/LatestFolder/SR0.0_0.5_1_10000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0:50:00
#SBATCH --mem-per-cpu=2100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py ozstar_pp_check_1e4_8 1 10000 0.0 0.5
srun python3 calculateirfvalues.py ozstar_pp_check_1e4_8 1 1 101 161 32