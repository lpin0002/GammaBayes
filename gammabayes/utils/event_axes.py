
from gammabayes.priors.CCR_bkg import log_bkg_CCR_dist, bkgfull
from gammabayes.utils.utils import hdp_credible_interval_1d
from gammabayes.utils.integration import logspace_riemann
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.interpolate import interp1d





def makelogjacob(energyaxis: np.ndarray) -> np.ndarray:
    """Generates log jacobian for using log-spaced energy for proper integrals

    Args:
        log10eaxis (np.ndarray, optional): Axis of discrete values of energy values. 

    Returns:
        np.ndarray: Log jacobian for using log-spaced energy for proper integrals
    """
    outputlogjacob = np.log(energyaxis)
    return outputlogjacob



def create_linear_axis(lower_bound: float, upper_bound: float,resolution: int = 10) -> np.ndarray:
    return np.linspace(lower_bound, 
                        upper_bound, 
                        int(round((upper_bound-lower_bound)/resolution)+1))
    

def create_loguniform_axis(lower_bound: float, upper_bound: float, number_of_bins_per_unit: int = 100) -> np.ndarray:
    return np.logspace(np.log10(lower_bound), 
                        np.log10(upper_bound), 
                        int(round(number_of_bins_per_unit*(np.log10(upper_bound)-np.log10(lower_bound)))+1))
    
def create_axes(energy_min: float, energy_max: float, 
                    energy_bins_per_decade: int, spatial_res: float, 
                    longitude_min: float, longitude_max: float,
                    latitude_min: float, latitude_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    print(energy_min, energy_max, 
                    energy_bins_per_decade, spatial_res, 
                    longitude_min, longitude_max,
                    latitude_min, latitude_max)
    
    energy_axis = create_loguniform_axis(energy_min, energy_max, number_of_bins_per_unit=energy_bins_per_decade)
    longitude_axis = create_linear_axis(longitude_min, longitude_max, resolution=spatial_res)
    latitude_axis = create_linear_axis(latitude_min, latitude_max, resolution=spatial_res)

    return energy_axis, longitude_axis, latitude_axis



def derive_edisp_bounds(irf_loglike, percentile=90, sigmalevel=5):
    energy_recon_axis, longitudeaxis, latitudeaxis = irf_loglike.axes
    energy_true_axis, longitudeaxistrue, latitudeaxistrue = irf_loglike.dependent_axes
    # Calc Energy Dispersion Bounds
    lonval = np.max(longitudeaxistrue)
    latval = np.max(latitudeaxistrue)
    edisp_val_bounds = []

    edisp_logval_bounds = []

    for erecon in tqdm(energy_recon_axis, desc='Calculating energy dispersion bounds'):
        log_edisp_vals = irf_loglike.log_edisp(energy_true_axis*0+erecon,energy_true_axis , 0*energy_true_axis+lonval, 0*energy_true_axis+latval)
        log_edisp_vals = log_edisp_vals - logspace_riemann(logy=log_edisp_vals, x=energy_true_axis)
        edisp_vals = np.exp(log_edisp_vals)
        bounds = hdp_credible_interval_1d(y=edisp_vals, sigma=sigmalevel, x=energy_true_axis)
        valdiff = np.diff(bounds)[0]
        logvaldiff = np.diff(np.log10(bounds))[0]
        edisp_val_bounds.append(valdiff)
        edisp_logval_bounds.append(logvaldiff)

    edisp_val_bound = np.percentile(edisp_val_bounds, percentile)
    edisp_logval_bound = np.percentile(edisp_logval_bounds, percentile)

    # The results are divided by two due to usage similar to radii, 
        # while what I have calculated is closer to diameter
    return edisp_val_bound/2, edisp_logval_bound/2
    


def derive_psf_bounds(irf_loglike, 
                      percentile=50, sigmalevel: int=6, 
                      axis_buffer: int = 4, parameter_buffer:float = 1.5, 
                      default_etrue_val: float = 0.2, n: int = 1000):
    longitudeaxis, latitudeaxis = irf_loglike.axes
    longitudeaxistrue, latitudeaxistrue = irf_loglike.dependent_axes

    fig1, ax = plt.subplots(1,1)
    radii = []
    # levels=(1 - np.exp(-8), 1 - np.exp(-25/2), 1 - np.exp(-36/2)),
    levels=(1 - np.exp(-sigmalevel**2/2),),


    for _i, recon_lon in tqdm(enumerate(longitudeaxis[::axis_buffer]), 
                              total=longitudeaxis[::axis_buffer].size,
                              desc='Calculating PSF bounds'):
        for _j, recon_lat in enumerate(latitudeaxis[::axis_buffer]):
            temp_lon_axis = longitudeaxistrue[(longitudeaxistrue>recon_lon-parameter_buffer)&(longitudeaxistrue<recon_lon+parameter_buffer)]
            temp_lat_axis = latitudeaxistrue[(latitudeaxistrue>recon_lat-parameter_buffer)&(latitudeaxistrue<recon_lat+parameter_buffer)]

            meshes = np.asarray(np.meshgrid(temp_lon_axis, temp_lat_axis, indexing='ij'))

            log_psf_vals = irf_loglike.log_psf(0*meshes[0].flatten()+recon_lon,
                                                                    0*meshes[1].flatten()+recon_lat,
                                                                    meshes[0].flatten()*0+default_etrue_val,
                                                                    meshes[0].flatten(), meshes[1].flatten(), 
                                                                    ).reshape(meshes[0].shape)
            
            log_psf_vals = np.squeeze(log_psf_vals)

            log_psf_vals = log_psf_vals - logspace_riemann(logy=logspace_riemann(logy=log_psf_vals, x=temp_lat_axis, axis=1), x=temp_lon_axis, axis=0)

            normed_marginal_dist = np.exp(log_psf_vals - logsumexp(log_psf_vals)).T
            t = np.linspace(0, normed_marginal_dist.max(), n)
            integral = ((normed_marginal_dist >= t[:, None, None]) * normed_marginal_dist).sum(axis=(1,2))

            f = interp1d(integral, t)
            t_contours = np.asarray(f(levels))
            t_contours[0].sort()
            contour_set = ax.contour(temp_lon_axis, temp_lat_axis, normed_marginal_dist, levels=t_contours[0], colors='None', linewidths=0.5)


            # Find the position of the highest data value
            max_value_index = np.unravel_index(np.argmax(log_psf_vals, axis=None), log_psf_vals.shape)
            center_x = meshes[0][max_value_index]
            center_y = meshes[1][max_value_index]

            for idx, level in enumerate(contour_set.levels):
                segments = contour_set.allsegs[idx]
                try:
                    segments = np.asarray(segments)
                except:
                    segments = np.asarray(segments[0])

                distances = np.sqrt((segments[:,0] - center_x) ** 2 + (segments[:,1] - center_y) ** 2)

                # Calculate the standard deviation of these distances
                mean_dist = np.max(distances)
                radii.append(mean_dist)
            ax.cla()

    plt.close()

    radii = np.asarray(radii)
    radii.sort()

    return np.percentile(radii, percentile)

