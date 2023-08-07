#!/bin/bash
#
#SBATCH --job-name=SR0.5|0.5|1|3|ozstar_imaptest
#SBATCH --output=data/LatestFolder/SR0.5_0.5_1_1000_ozstar_imaptest.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:10:00
#SBATCH --mem-per-cpu=16000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py ozstar_imaptest 1 500 0.5 0.5
srun python3 nuisancemarginalisation.py ozstar_imaptest 1 2 101 8