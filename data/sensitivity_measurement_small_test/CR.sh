#!/bin/bash
#
#SBATCH --job-name=CR0.5|0.0|5|sensitivity_measurement_small_test
#SBATCH --output=data/LatestFolder/CR0.5_0.0_100000_sensitivity_measurement_small_test.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=3000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 gridsearch.py sensitivity_measurement_small_test 161 1