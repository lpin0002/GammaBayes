#!/bin/bash
#
#SBATCH --job-name=DM1.0_25_0.8_4
#SBATCH --output=data/LatestFolder/DM1.0_25_0.8_10000.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:10:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 marginalisationnested.py windowbigtest4 25 400 1.0 0.8 10 41 161 10000