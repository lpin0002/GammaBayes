from scipy import integrate, special, interpolate, stats
import numpy as np
from tqdm import tqdm
from gammapy.irf import load_cta_irfs
from astropy import units as u

from astropy.coordinates import SkyCoord
from gammapy.maps import Map, MapAxis, MapAxes, WcsGeom
import sys, yaml, pickle, os, random, time, warnings

from os import path
resources_dir = path.join(path.dirname(__file__), '../package_data')



np.seterr(divide = 'ignore')
# I believe this is the alpha configuration of the array as there are no LSTs
irfs = load_cta_irfs(resources_dir+'/Prod5-South-20deg-AverageAz-14MSTs37SSTs.180000s-v0.1.fits')




def convertlonlat_to_offset(fov_coord):
    # Currently assuming small angles (|angle|<=4)
    return np.linalg.norm(fov_coord, axis=0)


def angularseparation(coord1, coord2=None):
    # Currently assuming small angles (|angle|<=4)
    
    try:
        return np.linalg.norm(coord2-coord1, axis=0)
    except:
        try:
            return np.linalg.norm(coord2-coord1.T, axis=1)
        except:
            return np.linalg.norm(coord2.T-coord1, axis=1)

def angularseparation_quick(coord1, coord2=None):
    # Currently assuming small angles (|angle|<=4)
    
    return np.linalg.norm(coord2-coord1, axis=0)



edispfull = irfs['edisp']
psffull = irfs['psf']
edispfull.normalize()
bkgfull = irfs['bkg']
psf3d = psffull.to_psf3d()
aefffull = irfs['aeff']

offsetaxis = psf3d.axes['rad'].center.value

bkgfull2d = bkgfull.to_2d()
bkgfull2doffsetaxis = bkgfull2d.axes['offset'].center.value
offsetaxisresolution = bkgfull2doffsetaxis[1]-bkgfull2doffsetaxis[0] # Comes out to 0.2
latbound            = 3.
lonbound            = 3.5



latitudeaxis            = np.linspace(-latbound, latbound, int(round(2*latbound/0.4)))
latitudeaxistrue        = np.linspace(-latbound, latbound, int(round(2*latbound/0.2)))

longitudeaxis           = np.linspace(-lonbound, lonbound, int(round(2*lonbound/0.4))) 
longitudeaxistrue       = np.linspace(-lonbound, lonbound, int(round(2*lonbound/0.2))) 


# Restricting energy axis to values that could have non-zero or noisy energy dispersion (psf for energy) values
log10estart             = -1.0
log10eend               = 2.0
log10erange             = log10eend - log10estart
log10eaxis              = np.linspace(log10estart,log10eend,int(np.round(log10erange*50))+1)
log10eaxistrue          = np.linspace(log10estart,log10eend,int(np.round(log10erange*200))+1)



def makelogjacob(log10eaxis=log10eaxis):
    outputlogjacob = np.log(10**log10eaxis)#+np.log(np.log(10))+np.log(log10eaxis[1]-log10eaxis[0])
    return outputlogjacob

logjacob = makelogjacob(log10eaxis)
logjacobtrue = makelogjacob(log10eaxistrue)


def edisp(logereconstructed, logetrue, truespatialcoord):
    return np.log(edispfull.evaluate(energy_true=np.power(10.,logetrue)*u.TeV,
                                                    migra = np.power(10.,logereconstructed-logetrue), 
                                                    offset=convertlonlat_to_offset(truespatialcoord)*u.deg).value)
    
    
def edisp_test(reconloge, logetrue, true_lon, true_lat):
    return np.log(edispfull.evaluate(energy_true=np.power(10.,logetrue)*u.TeV,
                                                    migra = np.power(10.,reconloge-logetrue), 
                                                    offset=convertlonlat_to_offset(np.array([true_lon, true_lat]))*u.deg).value)

def edisp_efficient(logereconstructed, logetrue, offset):
    return np.log(edispfull.evaluate(energy_true=np.power(10.,logetrue)*u.TeV,
                                                    migra = np.power(10.,logereconstructed-logetrue), 
                                                    offset=offset*u.deg).value)
def aeff_efficient(logetrue, offset):
    return np.log(aefffull.evaluate(energy_true=10**logetrue*u.TeV, offset=offset*u.deg).to(u.cm**2).value)



def psf(reconstructed_spatialcoord, logetrue, truespatialcoord):
    
    rad = angularseparation(reconstructed_spatialcoord, truespatialcoord).flatten()
    offset  = convertlonlat_to_offset(truespatialcoord).flatten()
    energyvals = np.power(10.,logetrue.flatten())
    output = np.log(psffull.evaluate(energy_true=energyvals*u.TeV,
                                                    rad = rad*u.deg, 
                                                    offset=offset*u.deg).value)
    
    return output


def psf_test(recon_lon, recon_lat, logetrue, true_lon, true_lat):
    reconstructed_spatialcoord = np.array([recon_lon, recon_lat])
    truespatialcoord = np.array([true_lon, true_lat])
    rad = angularseparation(reconstructed_spatialcoord, truespatialcoord).flatten()
    offset  = convertlonlat_to_offset(truespatialcoord).flatten()
    output = np.log(psffull.evaluate(energy_true=10**logetrue*u.TeV,
                                                    rad = rad*u.deg, 
                                                    offset=offset*u.deg).value)
    
    return output


def single_likelihood(reconloge, recon_lon, recon_lat, logetrue, true_lon, true_lat):
    reconstructed_spatialcoord = np.array([recon_lon, recon_lat])
    truespatialcoord = np.array([true_lon, true_lat])
    rad = angularseparation(reconstructed_spatialcoord, truespatialcoord).flatten()
    offset  = convertlonlat_to_offset(truespatialcoord).flatten()
    output = np.log(psffull.evaluate(energy_true=10**logetrue*u.TeV,
                                                    rad = rad*u.deg, 
                                                    offset=offset*u.deg).value)
    
    output+=np.log(edispfull.evaluate(energy_true=10**logetrue*u.TeV,
                                                    migra = 10**(reconloge-logetrue), 
                                                    offset=offset*u.deg).value)
    
    return output

def psf_efficient(rad, logetrue, offset):

    output = np.log(psffull.evaluate(energy_true=10**logetrue*u.TeV,
                                                    rad = rad*u.deg, 
                                                    offset=offset*u.deg).value)
    
    return output




def bkgdist(logeval, lon, lat):
    # np.log(1e6) factor is because the background rate is given in 1/MeV not 1/TeV for some reason
    return np.log(bkgfull.evaluate(energy=10**logeval*u.TeV, fov_lon=np.abs(lon)*u.deg, fov_lat=np.abs(lat)*u.deg).value*1e6)

# Does not have any mention of the log of the jacobian to keep it more general.
def inverse_transform_sampling(logpmf, Nsamples=1):
    
    logpmf = logpmf - special.logsumexp(logpmf)
    logcdf = np.logaddexp.accumulate(logpmf)
    cdf = np.exp(logcdf-logcdf[-1])  

    randvals = [random.random() for xkcd in range(Nsamples)]
    indices = [np.searchsorted(cdf, u) for u in randvals]
    return indices





import matplotlib.transforms as transforms
import matplotlib.patches as patches

def confidence_ellipse(x, y, probabilities, ax, n_std=3.0, edgecolor='white',facecolor='none', **kwargs):
    
    if not isinstance(probabilities, np.ndarray):
        probabilities = np.array(probabilities)

    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a 2-dimensional array")

    n, m = probabilities.shape

    
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()

    cov = np.cov(x, y, aweights=probabilities.flatten())
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = patches.Ellipse((0, 0),
                              width=ell_radius_x * 2,
                              height=ell_radius_y * 2,
                              edgecolor = edgecolor,
                              facecolor = facecolor,
                              **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.sum(x * probabilities.flatten()) / np.sum(probabilities)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.sum(y * probabilities.flatten()) / np.sum(probabilities)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def read_config_file(file_path):
    print(f"config file path: {file_path}")
    try:
        with open(file_path, 'r') as file:
            inputs = yaml.safe_load(file)
        return inputs
    except FileNotFoundError:
        print(f"Error: Input file '{file_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError:
        print(f"Error: Unable to parse YAML in '{file_path}'. Please ensure it is valid yaml file.")
        sys.exit(1)

def load_hyperparameter_pickle(file_path):
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


def save_config_file(config_dict, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False)
    print("Configuration saved to config_dict")

        
        
def check_necessary_config_inputs(input_dict):
    
    print('\n\n')

    try:
        print(f"Number of events for this script is {input_dict['Nevents']}")
    except:
        raise Exception("Number of events to be simulated/analysed not provided. Add line 'Nevents:  [number value]' to yaml file")
    
    try:
        print(f"Provided log10 mass is {input_dict['logmass']}")
    except:
        raise Exception("Log10 mass not provided. Add line 'logmass:  [logmass value]' to yaml file")
    
    try:
        print(f"Provided signal fraction/xi is {input_dict['xi']}")
    except:
        raise Exception("Signal fraction not provided. Add line 'xi:  [fractional value]' to yaml file")
    
    try:
        print(f"Provided identifier is {input_dict['identifier']}")
    except:
        warnings.warn("Identifier not provided. Default value of date in format [yr]_[month]_[day]_[hour]_[minute] will be used.", UserWarning)
        input_dict['identifier'] = time.strftime("%y_%m_%d_%H_%M")
        
    try:
        print(f"Number of mass bins to be tested is {input_dict['nbins_logmass']}")
    except:
        warnings.warn("Number of mass bins not provided. Default value of 61 to be used.", UserWarning)
        input_dict['nbins_logmass'] = 61
        
    try:
        print(f"Number of signal fraction bins to be tested is {input_dict['nbins_xi']}")
    except:
        warnings.warn("Number of signal fraction bins not provided. Default value of 101 to be used.", UserWarning)
        input_dict['nbins_xi'] = 101
        
        
    try:
        print(f"Dark matter density profile is {input_dict['dmdensity_profile']}")
    except:
        warnings.warn("Dark matter density profile not provided. Default value of 'einasto' will be used.", UserWarning)
        input_dict['dmdensity_profile'] = 'einasto'
        
    try:
        print(f"Number of cores to be used is {input_dict['numcores']}")
    except:
        input_dict['numcores'] = 1
        
    try:
        print(f"Run number is {input_dict['runnumber']}")
    except:
        input_dict['runnumber'] = 1
        
    try:
        print(f"Total number of events is {input_dict['totalevents']}")
    except:
        input_dict['totalevents'] = input_dict['Nevents']
        
    print('\n\n')
        
