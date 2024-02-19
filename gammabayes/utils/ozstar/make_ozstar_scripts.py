
import os, sys, numpy as np, time, math, yaml

def makejobscripts(ozstar_config_dict:dict, job_config_file_path:str, path_to_run_file: str) -> str:

    if int(ozstar_config_dict['time_mins'])<10:
        ozstar_config_dict['time_mins'] = f"0{int(ozstar_config_dict['time_mins'])}"
    if bool(ozstar_config_dict['mail_progress']):
        mail_str = f"""#SBATCH --mail-type=ALL
#SBATCH --mail-user={ozstar_config_dict['mail_address']}"""

    job_str = f"""#!/bin/bash
#
#SBATCH --job-name={ozstar_config_dict['jobname']}
#SBATCH --output=data/LatestFolder/{ozstar_config_dict['jobname']}.txt
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={ozstar_config_dict['numcores']}
#SBATCH --time={ozstar_config_dict['time_hrs']}:{ozstar_config_dict['time_mins']}:00
#SBATCH --mem-per-cpu={ozstar_config_dict['mem_per_cpu']}
{mail_str}
source ~/.bashrc
srun python3 {path_to_run_file} {job_config_file_path}"""
    
    return job_str




