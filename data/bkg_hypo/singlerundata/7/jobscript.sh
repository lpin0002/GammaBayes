#!/bin/bash
#
#SBATCH --job-name=SR1.0|0.5|7|4|bkg_hypo
#SBATCH --output=data/LatestFolder/SR1.0_0.5_7_10000_bkg_hypo.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=3000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
conda init bash
conda activate DMPipe
srun python3 single_script_code.py /fred/oz233/lpinchbe/GammaBayes/data/bkg_hypo/singlerundata/7/inputconfig.yaml