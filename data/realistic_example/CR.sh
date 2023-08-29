#!/bin/bash
#
#SBATCH --job-name=CR0.5|0.0001|7|realistic_example
#SBATCH --output=data/LatestFolder/CR0.5_0.0001_10000000_realistic_example.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=10000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 gridsearch.py realistic_example 161 1