#!/bin/bash
#
#SBATCH --job-name=PipelineSetup
#SBATCH --output=data/LatestFolder/SetupOutput.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=32000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=progressemail1999@gmail.com
source activate DMPipe
srun python3 gammabayes/default_file_setup.py 1 1 1
