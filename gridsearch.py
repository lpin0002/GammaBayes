from utils3d import *
import numpy as np
from scipy import special
from tqdm import tqdm
import os, sys, time, functools
from multiprocessing import Pool, freeze_support
sys.path.append("BFCalc")


if __name__=="__main__":
    

    try:
        identifier = sys.argv[1]
    except:
        raise Exception("You have not input a valid identifier. Please review the name of the stemfolder for your runs.")
    
    try:
        numcores = int(sys.argv[2])
    except:
        numcores = 8
        
    lontrue_mesh_nuisance, logetrue_mesh_nuisance, lattrue_mesh_nuisance = np.meshgrid(longitudeaxistrue, log10eaxistrue, latitudeaxistrue)


    
    
    
    #Generating various paths to files within stem directory
    currentdirecyory = os.getcwd()
    stemdirectory = currentdirecyory+f'/data/{identifier}'
    print("\nstem directory: ", stemdirectory, '\n')
    
    rundirs = [x[0] for x in os.walk(stemdirectory)][1:]
    print("number of run directories: ", len(rundirs), '\n')
    
    # Extracting the starting 
    signal_log_marginalisationvalues = np.load(rundirs[0]+'/log_signal_marginalisations.npy')
    bkg_log_marginalisationvalues = np.load(rundirs[0]+'/log_background_marginalisations.npy')
    
    lambdarange = np.load(rundirs[0]+'/lambdarange_direct.npy')
    logmassrange = np.load(rundirs[0]+'/logmassrange_direct.npy')

    
    totalevents = int(np.load(f"{rundirs[0]}/params.npy")[1,1])
    eventaxis = np.where(np.array(list(signal_log_marginalisationvalues.shape))==totalevents)[0][0]
    print('event axis: ', eventaxis, list(signal_log_marginalisationvalues.shape), totalevents)
    
    for rundir in rundirs[1:]:
        totalevents += int(np.load(f"{rundir}/params.npy")[1,1])
    
    print("total events ", totalevents)
    
    for rundir in rundirs[1:]:
        signal_log_marginalisationvalues_singlerun = np.load(rundir+'/log_signal_marginalisations.npy')
        bkg_log_marginalisationvalues_singlerun = np.load(rundir+'/log_background_marginalisations.npy')
        signal_log_marginalisationvalues = np.append(signal_log_marginalisationvalues, signal_log_marginalisationvalues_singlerun, axis=eventaxis)
        bkg_log_marginalisationvalues = np.append(bkg_log_marginalisationvalues, bkg_log_marginalisationvalues_singlerun)
    
    
    firstrundirectory = rundirs[0]
            
    unnormalised_log_posterior = []

    for lambdaval in tqdm(lambdarange, total=lambdarange.shape[0]):
        unnormalised_log_posterior.append([np.sum(np.logaddexp(np.log(lambdaval)+signal_log_marginalisationvalues[logmassindex,:], np.log(1-lambdaval)+bkg_log_marginalisationvalues)) for logmassindex in range(len(list(logmassrange)))])


    unnormalised_log_posterior = np.array(unnormalised_log_posterior)
    print(time.strftime("Current time is %d of %b, at %H:%M:%S"))


    print(f"Shape of the array containing the log of the unnormalised posterior: {unnormalised_log_posterior.shape}")

    
    
    
    np.save(f'{stemdirectory}/lambdarange_direct.npy',lambdarange)
    np.save(f'{stemdirectory}/logmassrange_direct.npy',logmassrange)
    np.save(f'{stemdirectory}/unnormalised_logposterior_direct.npy', unnormalised_log_posterior)
    






