
import sys, numpy as np, os, time
sys.path.append("..")
from gammabayes.utils.config_utils import read_config_file, create_true_axes_from_config, create_recon_axes_from_config

from gammabayes.priors.prior import discrete_logprior
from gammabayes.likelihoods.likelihood import discrete_loglikelihood
from gammabayes.hyper_inference.hyperparameter_likelihood import hyperparameter_likelihood

from gammabayes.dark_matter.SS_DM_Construct import SS_DM_dist
from gammabayes.utils.utils import single_likelihood
from gammabayes.utils.utils import makelogjacob
from gammabayes.utils.utils import bkgdist
from gammabayes.utils.load_package_defaults import astrophysicalbackground
from scipy import special, interpolate
from gammabayes.utils.default_file_setup import default_file_setup
from gammabayes.utils.utils import edisp_test, psf_test
from tqdm import tqdm


sys.path.append("..")
sys.path.append("../gammabayes")
sys.path.append("../gammabayes/dark_matter/")





import functools, multiprocessing, yaml
from multiprocessing import Pool, freeze_support
import pandas as pd


config_file_path = sys.argv[1]

config_inputs = read_config_file(config_file_path)

log10_eaxis_true, longitude_axis_true, latitude_axis_true = create_true_axes_from_config(config_inputs)
log10_mass = config_inputs['logmass']

print(log10_eaxis_true.shape, longitude_axis_true.shape, latitude_axis_true.shape)

log10_eaxis, longitude_axis, latitude_axis = create_recon_axes_from_config(config_inputs)

stemfolder = f"data/{config_inputs['identifier']}"
stemdatafolder = stemfolder+"/singlerundata"
datafolder = stemdatafolder+f"/{config_inputs['runnumber']}"

os.makedirs('data', exist_ok=True)
os.makedirs(f"data/{config_inputs['identifier']}", exist_ok=True)
os.makedirs(stemdatafolder, exist_ok=True)





if config_inputs['batch_job']:
    os.makedirs(datafolder, exist_ok=True)
    psfnormalisationvalues = np.load(f"{stemdatafolder}/psfnormalisation.npy")
    edispnormalisationvalues = np.load(f"{stemdatafolder}/edispnormalisation.npy")

else:
    os.makedirs(datafolder, exist_ok=False)
    psfnormalisationvalues, edispnormalisationvalues = default_file_setup(setup_astrobkg=0, setup_irfnormalisations=1,
                                log10eaxistrue=log10_eaxis_true, longitudeaxistrue=longitude_axis_true, latitudeaxistrue=latitude_axis_true, 
                                save_results=False, outputresults=True)







nsig                = int(round(config_inputs['xi']*config_inputs['Nevents']))
nbkg                = int(round((1-config_inputs['xi'])*config_inputs['Nevents']))







SS_DM_dist_instance= SS_DM_dist(longitude_axis_true, latitude_axis_true)#, density_profile=config_inputs['dmdensity_profile'])
logDMpriorfunc = SS_DM_dist_instance.func_setup()


logjacob_true = np.meshgrid(makelogjacob(log10_eaxis_true), longitude_axis_true, latitude_axis_true, indexing='ij')[0]

DM_prior = discrete_logprior(logfunction=logDMpriorfunc, name='Scalar Singlet Dark Matter Prior',
                               axes=(log10_eaxis_true, longitude_axis_true, latitude_axis_true,), 
                               axes_names=['log10 energy', 'lon', 'lat'],
                               default_hyperparameter_values=(log10_mass,), 
                               hyperparameter_names=['mass'], logjacob=logjacob_true)




nuisancemesh = np.meshgrid(log10_eaxis_true, longitude_axis_true, latitude_axis_true, indexing='ij')

astrophysicalbackground = default_file_setup(setup_astrobkg=1, setup_irfnormalisations=0,
                                log10eaxistrue=log10_eaxis_true, longitudeaxistrue=longitude_axis_true, latitudeaxistrue=latitude_axis_true, 
                                save_results=False, outputresults=True)
unnormed_logbkgpriorvalues = np.logaddexp(np.squeeze(bkgdist(*nuisancemesh)),np.log(astrophysicalbackground))


logbkgpriorvalues = unnormed_logbkgpriorvalues - special.logsumexp(unnormed_logbkgpriorvalues+logjacob_true)





unnormed_logbkgpriorvalues = np.logaddexp(np.squeeze(bkgdist(*nuisancemesh)),np.log(astrophysicalbackground))


logbkgfunc_annoying = interpolate.RegularGridInterpolator((log10_eaxis_true, longitude_axis_true, latitude_axis_true,), np.exp(unnormed_logbkgpriorvalues))
logbkgfunc = lambda logenergy, longitude, latitude: np.log(logbkgfunc_annoying((logenergy, longitude, latitude)))


bkg_prior = discrete_logprior(logfunction=logbkgfunc, name='Background Prior',
                               axes=(log10_eaxis_true, longitude_axis_true, latitude_axis_true,), 
                               axes_names=['log10 energy', 'lon', 'lat'], logjacob=logjacob_true)


logjacob = makelogjacob(log10_eaxis)

edisp_like = discrete_loglikelihood(logfunction=edisp_test, 
                                    axes=(log10_eaxis,), axes_names='log10E recon',
                                    name='energy dispersion',
                                    dependent_axes=(log10_eaxis_true, longitude_axis_true, latitude_axis_true,), logjacob=logjacob,
                                    dependent_axes_names = ['log10E true', 'lon', 'lat'])

psf_like = discrete_loglikelihood(logfunction=psf_test, 
                                    axes=(longitude_axis, latitude_axis), axes_names=['longitude recon', 'latitude recon'],
                                    name='point spread function ',
                                    dependent_axes=(log10_eaxis_true, longitude_axis_true, latitude_axis_true,),
                                    dependent_axes_names = ['log10E true', 'lon', 'lat'])


nsig = int(round(config_inputs['xi']*config_inputs['Nevents']))
nbkg = int(round((1-config_inputs['xi'])*config_inputs['Nevents']))

siglogevals,siglonvals,siglatvals  = DM_prior.sample(nsig)

bkglogevals,bkglonvals,bkglatvals  = bkg_prior.sample(nbkg)



signal_log10e_measured = [np.squeeze(edisp_like.sample((logeval,*coord,), numsamples=1)) for logeval,coord  in tqdm(zip(siglogevals, np.array([siglonvals, siglatvals]).T), total=nsig)]
bkg_log10e_measured = [np.squeeze(edisp_like.sample((logeval,*coord,), numsamples=1)) for logeval,coord  in tqdm(zip(bkglogevals, np.array([bkglonvals, bkglatvals]).T), total=nbkg)]


sig_lonlat_psf_samples =  [psf_like.sample((logeval,*coord,), 1).tolist() for logeval,coord  in tqdm(zip(siglogevals, np.array([siglonvals, siglatvals]).T), total=nsig)]
bkg_lonlat_psf_samples =  [psf_like.sample((logeval,*coord,), 1).tolist() for logeval,coord  in tqdm(zip(bkglogevals, np.array([bkglonvals, bkglatvals]).T), total=nbkg)]




bkg_lon_measured = [bkg_coord[0] for bkg_coord in bkg_lonlat_psf_samples]
bkg_lat_measured = [bkg_coord[1] for bkg_coord in bkg_lonlat_psf_samples]
signal_lon_measured = [sig_coord[0] for sig_coord in sig_lonlat_psf_samples]
signal_lat_measured = [sig_coord[1] for sig_coord in sig_lonlat_psf_samples]

np.save(f'{datafolder}/recon_bkg_samples.npy', np.array([bkg_log10e_measured, bkg_lon_measured, bkg_lat_measured]))
np.save(f'{datafolder}/recon_sig_samples.npy', np.array([signal_log10e_measured, signal_lon_measured, signal_lat_measured]))

np.save(f'{datafolder}/true_bkg_samples.npy', np.array([bkglogevals,bkglonvals,bkglatvals]))
np.save(f'{datafolder}/true_sig_samples.npy', np.array([siglogevals,siglonvals,siglatvals]))

############################################################################################################
############################################################################################################
#####################            Analysis
############################################################################################################
############################################################################################################

startertimer = time.perf_counter()
print(startertimer)





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


logmasswindowwidth      = 12/np.sqrt(config_inputs['xi']*config_inputs['totalevents'])

logmasslowerbound       = config_inputs['logmass']-logmasswindowwidth
logmassupperbound       = config_inputs['logmass']+logmasswindowwidth

# if 1:
if logmasslowerbound<log10_eaxis_true[0]:
    logmasslowerbound = log10_eaxis_true[0]
# if 1:
if logmassupperbound>2:
    logmassupperbound = 2


logmassrange            = np.linspace(logmasslowerbound, logmassupperbound, config_inputs['nbins_logmass']) 

# %%
hyperparameter_likelihood_instance = hyperparameter_likelihood(
                                                               priors=(DM_prior, bkg_prior,), likelihood=single_likelihood, 
                                                               dependent_axes=(log10_eaxis_true,  longitude_axis_true, latitude_axis_true), 
                                                               dependent_logjacob=logjacob_true,
                                                               hyperparameter_axes = ((logmassrange,), (None,)), numcores=config_inputs['numcores'], likelihoodnormalisation = psfnormalisationvalues+edispnormalisationvalues)


measured_log10e = [float(measured_log10e_val) for measured_log10e_val in measured_log10e]
margresults = hyperparameter_likelihood_instance.nuisance_log_marginalisation(
    axisvals= (measured_log10e, measured_lon, measured_lat),)

hyperparameter_likelihood_instance.save_data(directory_path=datafolder)




endertimer = time.perf_counter()
runtime_seconds = endertimer-startertimer
time_diff_hours = runtime_seconds//(60*60)
time_diff_minutes = (runtime_seconds- 60*60*time_diff_hours)//60
timediff_seconds = int(round(runtime_seconds- 60*60*time_diff_hours-60*time_diff_minutes))


print(f"\n\nTime to run script:{time_diff_hours} hours, {time_diff_minutes} minutes and {timediff_seconds} seconds\n")