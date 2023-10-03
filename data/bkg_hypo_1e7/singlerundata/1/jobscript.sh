#!/bin/bash
#
#SBATCH --job-name=SR1.0|0.0005|1|7|bkg_hypo_1e7
#SBATCH --output=data/LatestFolder/SR1.0_0.0005_1_10000000_bkg_hypo_1e7.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:30:00
#SBATCH --mem-per-cpu=1400
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
conda init bash
conda activate DMPipe
srun python3 single_script_code.py /fred/oz233/lpinchbe/GammaBayes/data/bkg_hypo_1e7/singlerundata/1/inputconfig.yaml