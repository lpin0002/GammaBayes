#!/bin/bash
#
#SBATCH --job-name=SR1.0|0.5|1|4|pleaseman
#SBATCH --output=data/LatestFolder/SR1.0_0.5_1_10000_pleaseman.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=200
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
conda init bash
conda activate DMPipe
srun python3 sim_and_nuisance_marg.py /Users/lpin0002/Desktop/temporaryfolder/GammaBayes/data/pleaseman/singlerundata/1/inputconfig.yaml