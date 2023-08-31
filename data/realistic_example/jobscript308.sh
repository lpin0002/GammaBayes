#!/bin/bash
#
#SBATCH --job-name=SR0.5|0.0001|308|7|realistic_example
#SBATCH --output=data/LatestFolder/SR0.5_0.0001_308_10000000_realistic_example.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --time=10:30:00
#SBATCH --mem-per-cpu=16800
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 simulation.py realistic_example 308 20000 0.5 0.0001
srun python3 nuisancemarginalisation.py realistic_example 308 500 51 7