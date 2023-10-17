import numpy as np
from astropy import units as u
from gammapy.astro.darkmatter import (
    profiles,
    JFactory
)
from astropy.coordinates import SkyCoord
from gammapy.maps import Map, MapAxis, MapAxes, WcsGeom
from scipy import interpolate
import pandas as pd
from gammabayes.utils.utils import log_aeff, convertlonlat_to_offset
from scipy.special import logsumexp
from os import path
DM_dir = path.join(path.dirname(__file__), '')


# SS_DM_dist(longitudeaxis, latitudeaxis, density_profile=profiles.EinastoProfile())
class SS_DM_dist(object):
    
    def __init__(self, longitudeaxis, latitudeaxis, density_profile=profiles.EinastoProfile(), ratios=False):
        """Initialise an SS_DM_dist class instance.

        Args:
            longitudeaxis (np.ndarray): Array of the galactic longitude values 
                to sample for the calculation of the different J-factor

            latitudeaxis (np.ndarray): Array of the galactic latitude values 
                to sample for the calculation of the different J-factor

            density_profile (_type_, optional): The density profile to be used 
                for the calculation of the differential J-factor. Must be of
                the same type as the profile contained in the 
                gamma.astro.darkmatter.profiles module as we use Gammapy to
                calculate our differential J-factors.

                Defaults to profiles.EinastoProfile().

            ratios (bool, optional): A bool representing whether one wants to use the input differential cross-sections
                or the annihilation __ratios__. Defaults to False.
        """
        self.longitudeaxis      = longitudeaxis
        self.latitudeaxis       = latitudeaxis
        self.density_profile    = density_profile
        self.ratios             = ratios
        
    
    
        darkSUSY_to_PPPC_converter = {
            "nuenue":"nu_e",
            "e+e-": "e",
            "numunumu":"nu_mu",
            "mu+mu-":"mu",
            'nutaunutau':"nu_tau",
            "tau+tau-":"tau",
            "cc": "c",
            "bb": "b",
            "tt": "t",
            "W+W-": "W",
            "ZZ": "Z",
            "gg": "g",
            "gammagamma": "gamma",
            "HH": "h",
        }


        darkSUSY_BFs_cleaned = pd.read_csv(f"{DM_dir}/dark_matter_data/darkSUSY_BFs/darkSUSY_BFs_cleaned.csv", delimiter=' ')

        darkSUSY_massvalues = darkSUSY_BFs_cleaned.iloc[:,1]/1e3

        darkSUSY_lambdavalues = darkSUSY_BFs_cleaned.iloc[:,2]

        channelfuncdictionary = {}

       
        log10xvals = np.load(f"{DM_dir}/dark_matter_data/griddata/log10xvals_massenergy_diffflux_grid.npy")
        massvalues = np.load(f"{DM_dir}/dark_matter_data/griddata/massvals_massenergy_diffflux_grid.npy")

        for darkSUSYchannel in list(darkSUSY_to_PPPC_converter.keys()):
            try:
                gammapychannel = darkSUSY_to_PPPC_converter[darkSUSYchannel]
                
                tempspectragrid = np.load(f"{DM_dir}/dark_matter_data/griddata/channel={gammapychannel}_massenergy_diffflux_grid.npy")
                
                flux_space_spectra_func = interpolate.RegularGridInterpolator((np.log10(massvalues/1e3), log10xvals), np.array(tempspectragrid), 
                                                                                        method='linear', bounds_error=False, fill_value=1e-3000)
                
                channelfuncdictionary[darkSUSYchannel] =  lambda inputs: np.log(flux_space_spectra_func(inputs))
            except:
                channelfuncdictionary[darkSUSYchannel] = lambda logmass, log10x: log10x*-np.inf

        self.channelfuncdictionary = channelfuncdictionary
        
        darkSUSY_BFs_cleaned_vals = darkSUSY_BFs_cleaned.to_numpy()[:,3:]
        if self.ratios:
            darkSUSY_BFs_cleaned_vals = np.exp(np.log(darkSUSY_BFs_cleaned_vals) - logsumexp(np.log(darkSUSY_BFs_cleaned_vals), axis=1)[:, np.newaxis])
            _flux_space_partial_sigmav_interpolator_dictionary = {channel: interpolate.LinearNDInterpolator((darkSUSY_massvalues, darkSUSY_lambdavalues),darkSUSY_BFs_cleaned_vals[:,idx]) for idx, channel in enumerate(list(darkSUSY_to_PPPC_converter.keys()))}

        else:
            _flux_space_partial_sigmav_interpolator_dictionary = {channel: interpolate.LinearNDInterpolator((darkSUSY_massvalues, darkSUSY_lambdavalues),darkSUSY_BFs_cleaned_vals[:,idx]) for idx, channel in enumerate(list(darkSUSY_to_PPPC_converter.keys()))}
        
        self.partial_sigmav_interpolator_dictionary = {channel: lambda mass, couplingval:np.log(flux_space_func(mass, couplingval)) for channel, flux_space_func in _flux_space_partial_sigmav_interpolator_dictionary.items()}
        self.profile = density_profile

        # Adopt standard values used in HESS
        profiles.DMProfile.DISTANCE_GC = 8.5 * u.kpc
        profiles.DMProfile.LOCAL_DENSITY = 0.39 * u.Unit("GeV / cm3")

        self.profile.scale_to_local_density()

        self.central_position = SkyCoord(0.0, 0.0, frame="galactic", unit="deg")
        self.geom = WcsGeom.create(skydir=self.central_position, 
                            binsz=(self.longitudeaxis[1]-self.longitudeaxis[0], self.latitudeaxis[1]-self.latitudeaxis[0]),
                            width=(self.longitudeaxis[-1]-self.longitudeaxis[0]+self.longitudeaxis[1]-self.longitudeaxis[0], 
                                   self.latitudeaxis[-1]-self.latitudeaxis[0]+self.latitudeaxis[1]-self.latitudeaxis[0]),
                            frame="galactic")


        jfactory = JFactory(
            geom=self.geom, profile=self.profile, distance=profiles.DMProfile.DISTANCE_GC
        )
        _diffjfact_array_with_units = jfactory.compute_differential_jfactor().to(u.Unit("TeV^2 / sr m^5"))

        self._diffjfact_array_units = _diffjfact_array_with_units.unit

        self.diffjfact_array = (_diffjfact_array_with_units.value).T

        self.diffJfactor_function = interpolate.RegularGridInterpolator((self.longitudeaxis, self.latitudeaxis), 
                                                                        self.diffjfact_array, 
                                                                        method='linear', 
                                                                        bounds_error=False, 
                                                                        fill_value=0)

        




    def nontrivial_coupling(self, logmass, logenergy, coupling=0.1, 
                            partial_sigmav_interpolator_dictionary=None, channelfuncdictionary=None):
        """Calculates Z_2 scalar singlet dark matter annihilation gamma-ray 
            spectra for a set of mass and coupling values.

        Args:
            logmass (float): Float value of log_10 of the dark matter mass in 
                TeV.

            logenergy (np.ndarray or float): Float values for log_10 gamma ray 
                energy values in TeV.

            coupling (float, optional): Value for the Higgs coupling. Defaults 
                to 0.1.

            partial_sigmav_interpolator_dictionary (dict, optional): A dictionary
                where the keys are the names of the dark matter annihilation final
                states as in DarkSUSY and the values being interpolation functions
                to calculate the partial annihilation cross-sections for the
                respective final states for a log_10 mass (TeV) and coupling 
                values. Defaults to None.

            channelfuncdictionary (dict, optional): A dictionary
                where the keys are the names of the dark matter annihilation final
                states as in DarkSUSY and the values being interpolation functions
                to calculate the spectral flux of gamma-rays for the respective final
                state. Defaults to None.

        Returns:
            np.ndarray: The total gamma-ray flux for the Z_2 Scalar Singlet 
                dark matter model
        """
        if partial_sigmav_interpolator_dictionary is None:
            partial_sigmav_interpolator_dictionary = self.partial_sigmav_interpolator_dictionary
            
        if channelfuncdictionary is None:
            channelfuncdictionary = self.channelfuncdictionary
        
        logspectra = -np.inf

        for channel in channelfuncdictionary.keys():
            logspectra = np.logaddexp(logspectra, partial_sigmav_interpolator_dictionary[channel](10**logmass, coupling)+channelfuncdictionary[channel]((logmass, logenergy-logmass)))
        
        return logspectra

    

    def func_setup(self):
        """A method that pumps out a function representing the natural log of 
            the flux of dark matter annihilation gamma rays for a given log 
            energy, sky position, log mass and higgs coupling value.
        """
    
        def DM_signal_dist(log10eval, lonval, latval, logmass, coupling=0.1):
            spectralvals = self.nontrivial_coupling(logmass.flatten(), log10eval.flatten()).reshape(log10eval.shape)

            spatialvals = np.log(self.diffJfactor_function((lonval.flatten(), latval.flatten()))).reshape(log10eval.shape)

            logaeffvals = log_aeff(log10eval.flatten(), lonval.flatten(), latval.flatten()).reshape(log10eval.shape)
                    
            logpdfvalues = spectralvals+spatialvals+logaeffvals
            
            return logpdfvalues
        
        return DM_signal_dist