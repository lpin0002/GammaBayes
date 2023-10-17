import sys, os, time


# Following is so the module can be run as a primary script
if __name__=="__main__":
    logmass = float(sys.argv[1])
    xi_true = float(sys.argv[2])  
    numberofruns = int(sys.argv[3])
    singlerunevents = int(sys.argv[4])
    numcores = int(sys.argv[5])
    num_marg_hours = int(sys.argv[6])
    num_marg_minutes = int(sys.argv[7])
    num_combine_hours = int(sys.argv[8])
    num_combine_minutes = int(sys.argv[9])
    identifier = sys.argv[10]
    try:
        nbins_logmass = int(sys.argv[11])
    except:
        nbins_logmass = 101
        
    try:
        nbins_xi = int(sys.argv[12])
    except:
        nbins_xi = 161
        
    
    try:
        marginalisation_mem = int(sys.argv[13])
    except:
        marginalisation_mem = 200
        
    try:
        combination_mem = int(sys.argv[14])
    except:
        combination_mem = 1000
    try:
        dmdensity_profile = sys.argv[15]
    except:
        dmdensity_profile = 'einasto'
        

    try:
        num_true_energy_bins_per_decade = int(sys.argv[16])
    except:
        num_true_energy_bins_per_decade = 300

    try:
        num_recon_energy_bins_per_decade = int(sys.argv[17])
    except:
        num_recon_energy_bins_per_decade = 50

    try:
        true_spatial_res = float(sys.argv[18])
    except:
        true_spatial_res = 0.2
    try:
        recon_spatial_res = float(sys.argv[19])
    except:
        recon_spatial_res = 0.4


    try:
        immediate_run = int(sys.argv[20])
    except:
        immediate_run = 1

        
    strboi = f"""
    #!/bin/bash
    #
    #SBATCH --job-name=RunSetup
    #SBATCH --output=data/LatestFolder/RunSetup{identifier}.txt
    #
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=1
    #SBATCH --time=1:00:00
    #SBATCH --mem-per-cpu=32000
    #SBATCH --mail-type=ALL
    #SBATCH --mail-user=progressemail1999@gmail.com
    conda init bash
    conda activate DMPipe
    python gammabayes/utils/GB_OzStar_Pipe/ozstarscriptgen.py {logmass} {xi_true} {numberofruns} {singlerunevents} {numcores} {num_marg_hours} {num_marg_minutes} {num_combine_hours} {num_combine_minutes} {identifier} {nbins_logmass} {nbins_xi} {marginalisation_mem} {combination_mem} {dmdensity_profile} {num_true_energy_bins_per_decade} {num_recon_energy_bins_per_decade} {true_spatial_res} {recon_spatial_res} {immediate_run}"""


    with open(f"data/start_jobscript.sh", 'w') as f:
                f.write(strboi)

    time.sleep(0.1)
    os.system(f"sbatch data/jobscript.sh")
    

#python gammabayes/utils/GB_OzStar_Pipe/ozstar_script_gen_jobscript.py 1.2 0.5 10 10000 16 1 30 0 30 new_file_sys_test 101 161 2000 32000 einasto 300 75 0.2 0.4