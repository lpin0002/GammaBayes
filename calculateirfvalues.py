from utils3d import *
import numpy as np
from scipy import special
from tqdm import tqdm
import os, sys
sys.path.append("BFCalc")


if __name__=="__main__":
    
    try:
        identifier = sys.argv[1]
    except:
        raise Exception("You have not input a valid identifier. Please review the name of the stemfolder for your runs.")
    
    try:
        numcores = int(sys.argv[2])
    except:
        numcores = 10
    
    
    
    
    currentdirecyory = os.getcwd()
    stemdirectory = currentdirecyory+f'/data/{identifier}'
    print("\nstem directory: ", stemdirectory, '\n')

    rundirs = [x[0] for x in os.walk(stemdirectory)][1:]
    print("number of run directories: ", len(rundirs), '\n')
    
    
    
    firstrundirectory = rundirs[0]
    
    print(firstrundirectory)
    
    
    signal_log10e_measured,  signal_lon_measured, signal_lat_measured = np.load(f"{firstrundirectory}/meassigsamples.npy")
    bkg_log10e_measured, bkg_lon_measured, bkg_lat_measured = np.load(f"{firstrundirectory}/measbkgsamples.npy")
    
    truelambda, Nsamples, truelogmassval = np.load(f"{firstrundirectory}/params.npy")[1,:]
    truelambda = float(truelambda)
    totalevents = int(Nsamples)
    
    logbkgpriorvalues = np.load(f"data/{identifier}/1/logbackgroundprior.npy")

    
    for rundir in rundirs[1:]:
        runnum = rundir.replace(stemdirectory+'/', '')
        print("runnum: ", runnum)
        params              = np.load(f"{rundir}/params.npy")
        signal_log10e_measured_temp, signal_lon_measured_temp, signal_lat_measured_temp = np.load(f"{rundir}/meassigsamples.npy")
        bkg_log10e_measured_temp, bkg_lon_measured_temp, bkg_lat_measured_temp = np.load(f"{rundir}/measbkgsamples.npy")
        

        
        signal_log10e_measured = np.concatenate((signal_log10e_measured, signal_log10e_measured_temp))
        signal_lon_measured = np.concatenate((signal_lon_measured, signal_lon_measured_temp))
        signal_lat_measured = np.concatenate((signal_lat_measured, signal_lat_measured_temp))

        bkg_log10e_measured = np.concatenate((bkg_log10e_measured, bkg_log10e_measured_temp))
        bkg_lon_measured = np.concatenate((bkg_lon_measured, bkg_lon_measured_temp))
        bkg_lat_measured = np.concatenate((bkg_lat_measured, bkg_lat_measured_temp))
        
        params[1,:]         = params[1,:]
        truelogmass         = float(params[1,2])
        nevents             = int(params[1,1])
        totalevents         +=nevents
        truelambdaval       = float(params[1,0])
            
    
        
    measured_log10e = list(signal_log10e_measured)+list(bkg_log10e_measured)
    measured_lon = list(signal_lon_measured)+list(bkg_lon_measured)
    measured_lat = list(signal_lat_measured)+list(bkg_lat_measured)


    psfnormalisation = np.load('psfnormalisation.npy')
    
    lontrue_mesh_edisp, logetrue_mesh_edisp, lattrue_mesh_edisp, logerecon_mesh_edisp,  = np.meshgrid(spatialaxistrue, log10eaxistrue, spatialaxistrue, log10eaxis)
    
    
    edispnormalisation = edisp(logerecon_mesh_edisp.flatten(), logetrue_mesh_edisp.flatten(), np.array([lontrue_mesh_edisp.flatten(), lattrue_mesh_edisp.flatten()])).reshape(logetrue_mesh_edisp.shape)
    edispnormalisation  = special.logsumexp(edispnormalisation+logjacob, axis=-1)




    edispnormalisation[edispnormalisation==-np.inf] = 0
    psfnormalisation[psfnormalisation==-np.inf] = 0   
    
    
    
    lontrue_mesh_nuisance, logetrue_mesh_nuisance, lattrue_mesh_nuisance = np.meshgrid(spatialaxistrue, log10eaxistrue, spatialaxistrue)

    irfproblist = []

    for logeval, coord in tqdm(zip(measured_log10e, np.array([measured_lon, measured_lat]).T), total=len(list(measured_log10e)), ncols=100, desc="Calculating IRF values"):
        irfproblist.append(psf(coord, np.array([lontrue_mesh_nuisance.flatten(), lattrue_mesh_nuisance.flatten()]), logetrue_mesh_nuisance.flatten()).reshape(logetrue_mesh_nuisance.shape)+\
            edisp(logeval, logetrue_mesh_nuisance.flatten(), np.array([lontrue_mesh_nuisance.flatten(), lattrue_mesh_nuisance.flatten()])).reshape(logetrue_mesh_nuisance.shape) - edispnormalisation - psfnormalisation)
        
    print(len(irfproblist))
    print(Nsamples)
    
    assert len(irfproblist) == int(Nsamples)
    assert irfproblist[0].shape == (log10eaxistrue.shape[0], spatialaxistrue.shape[0],spatialaxistrue.shape[0])
        
             
    np.save(f"{stemdirectory}/irfproblist.npy", irfproblist)