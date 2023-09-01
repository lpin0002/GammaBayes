#!/bin/bash
#
#SBATCH --job-name=SR0.0|0.5|2|4|re_ex_eres_200_test4
#SBATCH --output=data/LatestFolder/SR0.0_0.5_2_20000_re_ex_eres_200_test4.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=16:50:00
#SBATCH --mem-per-cpu=2400
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py re_ex_eres_200_test4 2 10000 0.0 0.5
srun python3 nuisancemarginalisation.py re_ex_eres_200_test4 2 2 51 24