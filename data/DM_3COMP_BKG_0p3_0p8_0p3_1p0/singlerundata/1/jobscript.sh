#!/bin/bash
#
#SBATCH --job-name=progressemail1999@gmail.com
#SBATCH --output=data/LatestFolder/SetupOutput.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=16000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
conda init bash
conda activate DMPipe
srun python3 gammabayes.standard_inference.Z3_DM_3COMP_BKG /Users/lpin0002/Desktop/temporaryfolder/GammaBayes/gammabayes/utils/ozstar/default_ozstar_config.yaml 