#!/bin/bash
#
#SBATCH --job-name=DM0.0_53_0.8_3
#SBATCH --output=data/LatestFolder/DM0.0_53_0.8_1000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=400
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py recoztest5 53 10 0.0 0.8 10