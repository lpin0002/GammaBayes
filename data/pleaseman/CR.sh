#!/bin/bash
#
#SBATCH --job-name=CR1.0|0.5|4|pleaseman
#SBATCH --output=data/LatestFolder/CR1.0_0.5_10000_pleaseman.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:00
#SBATCH --mem-per-cpu=1000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
conda init bash
conda activate DMPipe
srun python3 combine_results.py /Users/lpin0002/Desktop/temporaryfolder/GammaBayes/data/pleaseman/singlerundata/inputconfig.yaml