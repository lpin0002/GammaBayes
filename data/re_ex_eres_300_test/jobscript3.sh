#!/bin/bash
#
#SBATCH --job-name=SR0.0|0.0005|3|4|re_ex_eres_300_test
#SBATCH --output=data/LatestFolder/SR0.0_0.0005_3_10000_re_ex_eres_300_test.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:50:00
#SBATCH --mem-per-cpu=3100
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py re_ex_eres_300_test 3 1000 0.0 0.0005
srun python3 nuisancemarginalisation.py re_ex_eres_300_test 3 10 51 16