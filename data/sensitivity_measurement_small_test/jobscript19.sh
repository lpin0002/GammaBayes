#!/bin/bash
#
#SBATCH --job-name=SR0.5|0.0|19|5|sensitivity_measurement_small_test
#SBATCH --output=data/LatestFolder/SR0.5_0.0_19_100000_sensitivity_measurement_small_test.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --time=10:30:00
#SBATCH --mem-per-cpu=17000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py sensitivity_measurement_small_test 19 2000 0.5 0.0
srun python3 nuisancemarginalisation.py sensitivity_measurement_small_test 19 50 51 7