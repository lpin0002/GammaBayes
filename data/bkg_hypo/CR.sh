#!/bin/bash
#
#SBATCH --job-name=CR1.0|0.5|4|bkg_hypo
#SBATCH --output=data/LatestFolder/CR1.0_0.5_10000_bkg_hypo.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:30:00
#SBATCH --mem-per-cpu=16000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
conda init bash
conda activate DMPipe
srun python3 combine_results.py /fred/oz233/lpinchbe/GammaBayes/data/bkg_hypo/singlerundata/inputconfig.yaml