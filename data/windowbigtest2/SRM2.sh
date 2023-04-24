#!/bin/bash
#
#SBATCH --job-name=DM0.0_2_0.5_4
#SBATCH --output=data/LatestFolder/DM0.0_2_0.5_10000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:10:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py windowbigtest2 2 400 0.0 0.5 10 41 161 10000