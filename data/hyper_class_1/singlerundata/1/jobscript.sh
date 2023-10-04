#!/bin/bash
#
#SBATCH --job-name=SR1.0|0.05|1|5|hyper_class_1
#SBATCH --output=data/LatestFolder/SR1.0_0.05_1_100000_hyper_class_1.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:30:00
#SBATCH --mem-per-cpu=3000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
conda init bash
conda activate DMPipe
srun python3 single_script_code.py /fred/oz233/lpinchbe/GammaBayes/data/hyper_class_1/singlerundata/1/inputconfig.yaml