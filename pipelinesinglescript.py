from BFCalc.BFInterp import DM_spectrum_setup
from BFCalc.createspectragrids import singlechannel_diffflux, getspectrafunc, darkmatterdoubleinput, energymassinputspectralfunc
from utils3d import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
from astropy import units as u
from scipy import special,stats
from scipy.integrate import simps
from matplotlib import cm
from tqdm.autonotebook import tqdm as notebook_tqdm
import os, sys
import functools
from multiprocessing import Pool, freeze_support
import multiprocessing

sys.path.append("BFCalc")

if __name__=="__main__":

    astrophysicalbackground = np.load("unnormalised_astrophysicalbackground.npy")

    lonmeshtrue, log10emeshtrue, latmeshtrue = np.meshgrid(longitudeaxistrue, log10eaxistrue, latitudeaxistrue)
    latmeshrecon, lonmeshrecon = np.meshgrid(latitudeaxis, longitudeaxis)

    logjacobtrue = makelogjacob(log10eaxistrue)  

    print(lonmeshtrue.shape, lonmeshrecon.shape) 

    numberoftruevaluesamples = int(5e3)
    Nsamples=numberoftruevaluesamples
    truelambda          = 0.2
    nsig                = int(round(truelambda*Nsamples))
    nbkg                = int(round((1-truelambda)*Nsamples))

    truelambda          = nsig/(nbkg+nsig)
    numcores            = 1
    truelogmassval      = 0.0

    startertimer = time.perf_counter()
    print(startertimer)


    unnormed_logbkgpriorvalues = np.logaddexp(np.squeeze(bkgdist(log10emeshtrue, lonmeshtrue,latmeshtrue)),np.log(astrophysicalbackground))


    logbkgpriorvalues = unnormed_logbkgpriorvalues - special.logsumexp(unnormed_logbkgpriorvalues.T+logjacobtrue)
    logbkgpriorvalues = logbkgpriorvalues - special.logsumexp(logbkgpriorvalues.T+logjacobtrue)

    signalspecfunc = energymassinputspectralfunc
    signalfunc = setup_full_fake_signal_dist(truelogmassval, signalspecfunc)


    logsigpriorvalues = signalfunc(log10emeshtrue, lonmeshtrue,latmeshtrue)

    print(logsigpriorvalues.shape)

    logsigpriorvalues = (logsigpriorvalues - special.logsumexp(logsigpriorvalues.T+logjacobtrue))



    logsigbinnedprior = (logsigpriorvalues.T+logjacobtrue).T
    flattened_logsigbinnedprior = logsigbinnedprior.flatten()


    logbkgbinnedprior = (logbkgpriorvalues.T+logjacobtrue).T
    flattened_logbkgbinnedprior = logbkgbinnedprior.flatten()



    if truelambda!=0.0:
        sigresultindices = np.unravel_index(inverse_transform_sampling(flattened_logsigbinnedprior, Nsamples=nsig),logsigbinnedprior.shape)
        siglogevals = log10eaxistrue[sigresultindices[0]]
        siglonvals = longitudeaxistrue[sigresultindices[1]]
        siglatvals = latitudeaxistrue[sigresultindices[2]]
    else:
        siglogevals = np.asarray([])
        siglonvals = np.asarray([])
        siglatvals = np.asarray([])
        
        
        
    bkgresultindices = np.unravel_index(inverse_transform_sampling(flattened_logbkgbinnedprior, Nsamples=nbkg),logbkgbinnedprior.shape)
    bkglogevals = log10eaxistrue[bkgresultindices[0]]
    bkglonvals = longitudeaxistrue[bkgresultindices[1]]
    bkglatvals = latitudeaxistrue[bkgresultindices[2]]


    if truelambda!=0.0:
        signal_log10e_measured = log10eaxis[np.squeeze([inverse_transform_sampling(edisp(log10eaxis, logeval, coord)+logjacob, Nsamples=1) for logeval,coord  in notebook_tqdm(zip(siglogevals, np.array([siglonvals, siglatvals]).T), total=nsig)])]
    else:
        signal_log10e_measured = np.asarray([])
        
        
    if truelambda!=0:
        signal_spatial_indices = np.squeeze([inverse_transform_sampling(psf(np.array([lonmeshrecon.flatten(), latmeshrecon.flatten()]), coord, logeval).flatten(), Nsamples=1) for logeval, coord in notebook_tqdm(zip(siglogevals, np.array([siglonvals, siglatvals]).T), total=nsig)])
        signal_reshaped_indices = np.unravel_index(signal_spatial_indices, shape=lonmeshrecon.shape)
        signal_lon_measured = longitudeaxis[signal_reshaped_indices[0]]
        signal_lat_measured = latitudeaxis[signal_reshaped_indices[1]]
    else:
        signal_lon_measured = np.asarray([])
        signal_lat_measured = np.asarray([])
        
        
    bkg_log10e_measured = log10eaxis[np.squeeze([inverse_transform_sampling(edisp(log10eaxis, logeval, coord)+logjacob, Nsamples=1) for logeval,coord  in notebook_tqdm(zip(bkglogevals, np.array([bkglonvals, bkglatvals]).T), total=nbkg)])]


    bkg_spatial_indices = np.squeeze([inverse_transform_sampling(psf(np.array([lonmeshrecon.flatten(), latmeshrecon.flatten()]), coord, logeval).flatten(), Nsamples=1) for logeval, coord in notebook_tqdm(zip(bkglogevals, np.array([bkglonvals, bkglatvals]).T), total=nbkg)])
    bkg_reshaped_indices = np.unravel_index(bkg_spatial_indices, shape=lonmeshrecon.shape)
    bkg_lon_measured = longitudeaxis[bkg_reshaped_indices[0]]
    bkg_lat_measured = latitudeaxis[bkg_reshaped_indices[1]]


    try:
        measured_log10e = list(signal_log10e_measured)+list(bkg_log10e_measured)
        measured_lon = list(signal_lon_measured)+list(bkg_lon_measured)
        measured_lat = list(signal_lat_measured)+list(bkg_lat_measured)
        
    except:
        if type(bkg_log10e_measured)==np.float64:
            measured_log10e = list(signal_log10e_measured)
            measured_lon = list(signal_lon_measured)
            measured_lat = list(signal_lat_measured)
            measured_log10e.append(bkg_log10e_measured)
            measured_lon.append(bkg_lon_measured)
            measured_lat.append(bkg_lat_measured)
            
        elif type(signal_log10e_measured)==np.float64:
            measured_log10e = list(bkg_log10e_measured)
            measured_lon = list(bkg_lon_measured)
            measured_lat = list(bkg_lat_measured)
            measured_log10e.append(signal_log10e_measured)
            measured_lon.append(signal_lon_measured)
            measured_lat.append(signal_lat_measured)
        else:
            print('what')
            
            
            
    # Save the matrix of psf values in its entirety (not just normalisation values)?
    lontrue_mesh_psf, logetrue_mesh_psf, lattrue_mesh_psf, lonrecon_mesh_psf, latrecon_mesh_psf = np.meshgrid(longitudeaxistrue, log10eaxistrue, latitudeaxistrue, longitudeaxis, latitudeaxis)
    # psfmatrix = psf(np.array([lonrecon_mesh_psf.flatten(), latrecon_mesh_psf.flatten()]), np.array([lontrue_mesh_psf.flatten(), lattrue_mesh_psf.flatten()]), logetrue_mesh_psf.flatten()).reshape(logetrue_mesh_psf.shape)
    psfmatrix = np.load("psfmatrix.npy")


        
    lontrue_mesh_edisp, logetrue_mesh_edisp, lattrue_mesh_edisp, logerecon_mesh_edisp,  = np.meshgrid(longitudeaxistrue, log10eaxistrue, latitudeaxistrue, log10eaxis)


    # edispmatrix = edisp(logerecon_mesh_edisp.flatten(), logetrue_mesh_edisp.flatten(), np.array([lontrue_mesh_edisp.flatten(), lattrue_mesh_edisp.flatten()])).reshape(logetrue_mesh_edisp.shape)
    edispmatrix = np.load("edispmatrix.npy")



    psfnormalisation  = special.logsumexp(psfmatrix, axis=(-2,-1))

    edispnormalisation  = special.logsumexp(edispmatrix+logjacob, axis=-1)


    edispnormalisation[edispnormalisation==-np.inf] = 0
    psfnormalisation[psfnormalisation==-np.inf] = 0   

    edispmatrix = edispmatrix-edispnormalisation[:,:,:,np.newaxis]
    psfmatrix = psfmatrix-psfnormalisation[:,:,:,np.newaxis, np.newaxis]


    psfnormalisation  = special.logsumexp(psfmatrix, axis=(-2,-1))

    edispnormalisation  = special.logsumexp(edispmatrix+logjacob, axis=-1)


    edispnormalisation[edispnormalisation==-np.inf] = 0
    psfnormalisation[psfnormalisation==-np.inf] = 0   

    edispmatrix = edispmatrix-edispnormalisation[:,:,:,np.newaxis]
    psfmatrix = psfmatrix-psfnormalisation[:,:,:,np.newaxis, np.newaxis]

    lontrue_mesh_nuisance, logetrue_mesh_nuisance, lattrue_mesh_nuisance = np.meshgrid(longitudeaxistrue, log10eaxistrue, latitudeaxistrue)

    irfindexlist = []

        
    with Pool(numcores) as pool: 
        
            
        for result in notebook_tqdm(pool.imap(calcrirfindices, zip(measured_log10e, np.array([measured_lon, measured_lat]).T)), total=len(list(measured_log10e)), ncols=100, desc="Calculating irf values..."):
                irfindexlist.append(result)

        pool.close() 
        
    irfindexlist = np.array(irfindexlist)


    bkgpriorarray  = logbkgpriorvalues
    # bkgpriorarray = bkgpriorarray.T - special.logsumexp(bkgpriorarray.T+logjacobtrue)

    tempbkgmargfunc = functools.partial(marginalisenuisance, prior=bkgpriorarray, edispmatrix=edispmatrix, psfmatrix=psfmatrix)
    bkgmargvals = [tempbkgmargfunc(singleeventindices) for singleeventindices in tqdm(irfindexlist, total=irfindexlist.shape[0])]


    bkgmargvals = np.array(bkgmargvals)
    print(bkgmargvals.shape)



    sigdistsetup = setup_full_fake_signal_dist
    # Makes it so that when np.log(0) is called a warning isn't raised as well as other errors stemming from this.
    np.seterr(divide='ignore', invalid='ignore')





    nbinslogmass            = 61
    logmasswindowwidth      = 7/np.sqrt(nsig)


    logmasslowerbound       = truelogmassval-logmasswindowwidth
    logmassupperbound       = truelogmassval+logmasswindowwidth



    if logmasslowerbound<log10eaxis[0]:
        logmasslowerbound = log10eaxis[0]
    if logmassupperbound>2:
        logmassupperbound = 2


    logmassrange            = np.linspace(logmasslowerbound, logmassupperbound, nbinslogmass) 







    nbinslambda            = 101
    lambdawindowwidth      = 8/np.sqrt(Nsamples)


    lambdalowerbound       = truelambda-lambdawindowwidth
    lambdaupperbound       = truelambda+lambdawindowwidth



    if lambdalowerbound<0:
        lambdalowerbound = 0
    if lambdaupperbound>1:
        lambdaupperbound = 1


    lambdarange            = np.linspace(lambdalowerbound, lambdaupperbound, nbinslambda) 


    sigmarg_partial = functools.partial(sigmarg, specfunc=signalspecfunc, irfindexlist=irfindexlist, edispmatrix=edispmatrix, psfmatrix=psfmatrix,
                                                logetrue_mesh_nuisance=logetrue_mesh_nuisance, lontrue_mesh_nuisance=lontrue_mesh_nuisance, 
                                                lattrue_mesh_nuisance=lattrue_mesh_nuisance, logjacobtrue=logjacobtrue)
    print(time.strftime("Starting signal marginalisation, time is %d of %b, at %H:%M:%S"))

    with Pool(numcores) as pool:
        sigmargresults = pool.map(sigmarg_partial, tqdm(logmassrange, ncols=100, total=logmassrange.shape[0]))
    print(time.strftime("Finished signal marginalisation. Current time is %d of %b, at %H:%M:%S"))
    print("Now converting to numpy arrays and saving the result.")

    signal_log_marginalisationvalues = np.array(sigmargresults)



    log_posterior = []

    for lambdaval in notebook_tqdm(lambdarange, total=lambdarange.shape[0]):
        log_posterior.append([np.sum(np.logaddexp(np.log(lambdaval)+signal_log_marginalisationvalues[logmassindex,:], np.log(1-lambdaval)+bkgmargvals)) for logmassindex in range(len(list(logmassrange)))])

    log_posterior = np.array(log_posterior)-special.logsumexp(log_posterior)


    endertimer = time.perf_counter()
    print(endertimer-startertimer)


    from utils3d import confidence_ellipse
    from scipy.stats import norm

    import time

    colormap = cm.get_cmap('Blues_r', 4)

    fig, ax = plt.subplots(2,2, dpi=100, figsize=(10,8))
    plt.suptitle(f"Nevents= {Nsamples}", size=24)

    # Upper left plot
    logmass_logposterior = special.logsumexp(log_posterior, axis=0)

    normalisedlogmassposterior = np.exp(logmass_logposterior-special.logsumexp(logmass_logposterior))

    cdflogmassposterior = np.cumsum(normalisedlogmassposterior)
    mean = logmassrange[np.abs(norm.cdf(0)-cdflogmassposterior).argmin()]
    zscores = [-3, -2,-1,1,2, 3]
    logmasspercentiles = []
    for zscore in zscores:
        logmasspercentiles.append(logmassrange[np.abs(norm.cdf(zscore)-cdflogmassposterior).argmin()])


    ax[0,0].plot(logmassrange,normalisedlogmassposterior, c='tab:green')

    ax[0,0].axvline(mean, c='tab:green', ls=':')


    for o, percentile in enumerate(logmasspercentiles):
                color = colormap(np.abs(zscores[o])/4-0.01)

                ax[0,0].axvline(percentile, c=color, ls=':')
    ax[0,0].axvline(truelogmassval, ls='--', color="tab:orange")


    if min(mean - logmasspercentiles)>log10eaxistrue[1]-log10eaxistrue[0]:
        for logetrueval in log10eaxistrue:
            ax[0,0].axvline(logetrueval, c='forestgreen', alpha=0.3)
    ax[0,0].set_ylim([0, None])
    ax[0,0].set_xlim([logmassrange[0], logmassrange[-1]])

    # Upper right plot
    ax[0,1].axis('off')


    # Lower left plot
    # ax[1,0].pcolormesh(logmassrange, lambdarange, np.exp(normalisedlogposterior).T, cmap='Blues')
    ax[1,0].pcolormesh(logmassrange, lambdarange, np.exp(log_posterior), vmin=0)
    ax[1,0].axvline(truelogmassval, c='tab:orange')
    ax[1,0].axhline(truelambda, c='tab:orange')
    ax[1,0].set_xlabel(r'$log_{10}$ mass [TeV]')
    ax[1,0].set_ylabel(r'$\lambda$')

    ax[1,0].set_ylim([lambdarange[0], lambdarange[-1]])
    ax[1,0].set_xlim([logmassrange[0], logmassrange[-1]])

    ########################################################################################################################
    ########################################################################################################################
    # I have no clue how this works but I've checked it against some standard distributions and it seems correct
    normed_posterior = np.exp(log_posterior)/np.exp(log_posterior).sum()
    n = 100000
    t = np.linspace(0, normed_posterior.max(), n)
    integral = ((normed_posterior >= t[:, None, None]) * normed_posterior).sum(axis=(1,2))

    from scipy import interpolate
    f = interpolate.interp1d(integral, t)
    t_contours = f(np.array([1-np.exp(-4.5),1-np.exp(-2.0),1-np.exp(-0.5)]))
    ax[1,0].contour(normed_posterior, t_contours, extent=[logmassrange[0],logmassrange[-1], lambdarange[0],lambdarange[-1]], colors='white', linewidths=0.5)
    ########################################################################################################################
    ########################################################################################################################


    lambda_logposterior = special.logsumexp(log_posterior, axis=1)

    normalisedlambdaposterior = np.exp(lambda_logposterior-special.logsumexp(lambda_logposterior))

    cdflambdaposterior = np.cumsum(normalisedlambdaposterior)
    meanlambda = lambdarange[np.abs(norm.cdf(0)-cdflambdaposterior).argmin()]
    lambdapercentiles = []
    for zscore in zscores:
        lambdapercentile = lambdarange[np.abs(norm.cdf(zscore)-cdflambdaposterior).argmin()]
        lambdapercentiles.append(lambdapercentile)
        print(np.sqrt(1e5/1e8)*np.abs(lambdapercentile - meanlambda))





    ax[1,1].plot(lambdarange,normalisedlambdaposterior, c='tab:green')

    ax[1,1].axvline(meanlambda, c='tab:green', ls=':')


    for o, percentile in enumerate(lambdapercentiles):
                color = colormap(np.abs(zscores[o])/4-0.01)

                ax[1,1].axvline(percentile, c=color, ls=':')
    ax[1,1].axvline(truelambda, ls='--', color="tab:orange")
    ax[1,1].set_xlabel(r'$\lambda$')
    ax[1,1].set_ylim([0, None])


    plt.savefig(time.strftime(f"Figures/TestFigures/{Nsamples}events_lm{truelogmassval}_l{truelambda}_%m%d_%H%M.pdf"))
    plt.show()