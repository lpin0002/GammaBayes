from gammabayes.likelihoods.core import DiscreteLogLikelihood
from gammabayes.likelihoods.irfs.irf_extractor_class import IRFExtractor
from gammabayes.likelihoods.irfs.irf_normalisation_setup import irf_norm_setup
from astropy import units as u
from astropy.units import Quantity
import numpy as np
class IRF_LogLikelihood(DiscreteLogLikelihood):

    def __init__(self,
                 pointing_dir:list[Quantity]=[0*u.deg,0*u.deg], 
                 zenith:int=20, hemisphere:str='South', 
                 observation_time: float|u.Quantity = 50*u.hr,
                 prod_vers=5, 
                 instrument='CTAO',
                 obs_id = None,
                 psf_units: u.Unit = (1/u.deg**2).unit,
                 edisp_units: u.Unit = (1/u.TeV).unit,
                 aeff_units: u.Unit = u.cm**2,
                 CCR_BKG_units: u.Unit = (1/(u.deg**2*u.TeV*u.s)).unit,
                 *args, **kwargs):
        """_summary_

        Args:
            name (list[str] | tuple[str], optional): Identifier name(s) for the likelihood instance.

            pointing_dir (list, optional): Pointing direction of the telescope in galactic 
            coordinates (e.g. Directly pointing at the Galactic Centre is the default). Defaults to [0,0].

            zenith (int, optional): Zenith angle of the telescope (can be 20, 40 or 60 degrees). 
            Defaults to 20.

            hemisphere (str, optional): Which hemisphere the telescope observation was in, can be 'South' 
            or 'North'. Defaults to 'South'.

            prod_vers (int, optional): Version of the likelihood function, can currently be 3/3b or 5. 
            Defaults to 5.
            
            axes (list[np.ndarray] | tuple[np.ndarray]): Arrays defining the reconstructed observation value axes.
            
            dependent_axes (list[np.ndarray]): Arrays defining the true observation value axes.
            
            inputunit (str | list[str] | tuple[str], optional): Unit(s) of the input axes.
            
            axes_names (list[str] | tuple[str], optional): Names of the independent variable axes.
            
            dependent_axes_names (list[str] | tuple[str], optional): Names of the dependent variable axes.
            
            logspace_integrator (callable, optional): Integration method used for normalization.
            
            parameters (dict | ParameterSet, optional): Parameters for the log likelihood function.
        """
        self.irf_loglikelihood = IRFExtractor(zenith_angle=zenith, hemisphere=hemisphere, 
                                              prod_vers=prod_vers, 
                                              observation_time=observation_time, 
                                              instrument=instrument,
                                              pointing_dir=pointing_dir,
                                              obs_id=obs_id,
                                              psf_units = psf_units,
                                              edisp_units = edisp_units,
                                              aeff_units = aeff_units,
                                              CCR_BKG_units = CCR_BKG_units,
                                              )
        super().__init__(
            logfunction=self.irf_loglikelihood.single_loglikelihood, 
            *args, **kwargs
        )
        self.pointing_dir = pointing_dir


        self.psf_units = self.irf_loglikelihood.psf_units
        self.edisp_units = self.irf_loglikelihood.edisp_units
        self.aeff_units = self.irf_loglikelihood.aeff_units
        self.CCR_BKG_units = self.irf_loglikelihood.CCR_BKG_units


    def __call__(self,recon_energy, recon_lon, recon_lat, 
                 true_energy, true_lon, true_lat, 
                 pointing_dir=None, *args, **kwargs):
        """_summary_

        Args:
            recon_energy (float): Measured energy value by the CTA
            recon_lon (float): Measured FOV longitude of a gamma-ray event
                detected by the CTA
            recon_lat (float): Measured FOV latitude of a gamma-ray event
                detected by the CTA
            true_energy (float): True energy of a gamma-ray event detected by the CTA
            true_lon (float): True FOV longitude of a gamma-ray event 
                detected by the CTA
            true_lat (float): True FOV latitude of a gamma-ray event 
                detected by the CTA

        Returns:
            float: natural log of the full CTA likelihood for the given gamma-ray 
                event data
        """
        if pointing_dir is None:
            pointing_dir = self.pointing_dir
        
        return self.logfunction(recon_energy, recon_lon, recon_lat, 
                 true_energy, true_lon, true_lat,
                 pointing_dir=pointing_dir, *args, **kwargs)
    

    def log_edisp(self, recon_energy, true_energy, true_lon, true_lat, pointing_dir=None, *args, **kwargs):
        """
        Wrapper for the Gammapy interpretation of the CTA energy dispersion function.

        Args:
            recon_energy (Quantity): Measured energy value by the CTA.
            true_energy (Quantity): True energy of a gamma-ray event detected by the CTA.
            true_lon (Quantity): True FOV longitude of a gamma-ray event detected by the CTA.
            true_lat (Quantity): True FOV latitude of a gamma-ray event detected by the CTA.
            pointing_dir (list[Quantity], optional): Pointing direction. Defaults to [0*u.deg, 0*u.deg].

        Returns:
            float: Natural log of the CTA energy dispersion likelihood for the given gamma-ray event data.
        """
        if pointing_dir is None:
            pointing_dir = self.pointing_dir


        return self.irf_loglikelihood.log_edisp(recon_energy, true_energy, true_lon, true_lat, pointing_dir=pointing_dir, **kwargs, )
    
    def log_psf(self, recon_lon, recon_lat, true_energy, true_lon, true_lat, pointing_dir=None, **kwargs):
        """
        Wrapper for the Gammapy interpretation of the CTA point spread function.

        Args:
            recon_lon (Quantity): Measured FOV longitude of a gamma-ray event detected by the CTA.
            recon_lat (Quantity): Measured FOV latitude of a gamma-ray event detected by the CTA.
            true_energy (Quantity): True energy of a gamma-ray event detected by the CTA.
            true_lon (Quantity): True FOV longitude of a gamma-ray event detected by the CTA.
            true_lat (Quantity): True FOV latitude of a gamma-ray event detected by the CTA.
            pointing_dir (list[Quantity], optional): Pointing direction. Defaults to [0*u.deg, 0*u.deg].

        Returns:
            float: Natural log of the CTA point spread function likelihood for the given gamma-ray event data.
        """
        if pointing_dir is None:
            pointing_dir = self.pointing_dir
        
        return self.irf_loglikelihood.log_psf(recon_lon, recon_lat, true_energy, true_lon, true_lat, pointing_dir, **kwargs)
    
    def log_aeff(self, energy, lon, lat, *args, pointing_dir = None, **kwargs):
        """
        Wrapper for the Gammapy interpretation of the log of the CTA effective area function.

        Args:
            energy (Quantity): True energy of a gamma-ray event detected by the CTA.
            lon (Quantity): True FOV longitude of a gamma-ray event detected by the CTA.
            lat (Quantity): True FOV latitude of a gamma-ray event detected by the CTA.
            pointing_dir (list[Quantity], optional): Pointing direction. Defaults to [0*u.deg, 0*u.deg].

        Returns:
            float: The natural log of the effective area of the CTA in cm^2.
        """
        if pointing_dir is None:
            pointing_dir = self.pointing_dir

        return self.irf_loglikelihood.log_aeff(energy, lon, lat, *args, **kwargs, pointing_dir=self.pointing_dir, )
    

    def log_bkg_CCR(self, *args, pointing_dir=None, **kwargs):
        """
        Wrapper for the Gammapy interpretation of the log of the CTA's background charged cosmic-ray mis-identification rate.

        Args:
            energy (Quantity): True energy of a gamma-ray event detected by the CTA.
            longitude (Quantity): True FOV longitude of a gamma-ray event detected by the CTA.
            latitude (Quantity): True FOV latitude of a gamma-ray event detected by the CTA.
            spectral_parameters (dict, optional): Spectral parameters. Defaults to {}.
            spatial_parameters (dict, optional): Spatial parameters. Defaults to {}.
            pointing_dir (list[Quantity], optional): Pointing direction. Defaults to [0*u.deg, 0*u.deg].

        Returns:
            float: Natural log of the charged cosmic ray mis-identification rate for the CTA.
        """
        if pointing_dir is None:
            pointing_dir = self.pointing_dir

            
        return self.irf_loglikelihood.log_bkg_CCR(pointing_dir=pointing_dir, *args, **kwargs)


    def create_log_norm_matrices(self, **kwargs):
        """
        Creates normalization matrices for the IRF log likelihood.

        Args:
            **kwargs: Additional parameters for the normalization setup.

        Returns:
            dict: Normalization matrices for the log likelihood.
        """
        return irf_norm_setup(energy_true_axis=self.dependent_axes[0],
                            longitudeaxistrue=self.dependent_axes[1],
                            latitudeaxistrue=self.dependent_axes[2],

                            energy_recon_axis=self.axes[0],
                            longitudeaxis=self.axes[1],
                            latitudeaxis=self.axes[2],
                            
                            log_psf=self.log_psf,
                            log_edisp=self.log_edisp,
                            
                            **kwargs
                            )
    
    def to_dict(self):
        data_dict = {}
        data_dict["pointing_dir"] = self.pointing_dir
        data_dict["zenith"] = self.irf_loglikelihood.zenith
        data_dict["hemisphere"] = self.irf_loglikelihood.hemisphere
        data_dict["prod_vers"] = self.irf_loglikelihood.prod_vers
        data_dict["observation_time"] = self.irf_loglikelihood.observation_time
        data_dict["instrument"] = self.irf_loglikelihood.instrument

        return data_dict
    

    def peek(self, fig_kwargs={}, pcolormesh_kwargs={}, plot_kwargs={}, probability_scale='log', colormap='Blues', **kwargs):

        from matplotlib import pyplot as plt

        fig_kwargs.update(kwargs)

        plot_kwargs.setdefault('norm', probability_scale)
        pcolormesh_kwargs.setdefault('norm', probability_scale)


        plot_kwargs.setdefault('colormap', colormap)
        pcolormesh_kwargs.setdefault('cmap', colormap)

        fig_kwargs.setdefault('figsize', (12,6))

        fig, ax = plt.subplots(2, 3, **fig_kwargs)

        self._plot_edisp_pdf(ax=ax[0,0], plot_kwargs=plot_kwargs)
        self._plot_edisp_density(ax=ax[0,1], pcolormesh_kwargs=pcolormesh_kwargs)

        self._plot_psf_pdf(ax=ax[1,0], pcolormesh_kwargs=pcolormesh_kwargs)
        self._plot_psf_lines(ax=ax[1,1], plot_kwargs=plot_kwargs)

        self._plot_aeff(ax=ax[0,2], pcolormesh_kwargs=pcolormesh_kwargs)

        self._plot_ccr(ax=ax[1,2], plot_kwargs=plot_kwargs)



        plt.tight_layout()
        return fig, ax




    def _plot_edisp_pdf(self, ax=None, plot_kwargs={}):
        from gammabayes.core.utils import pick_5_values
        from matplotlib import cm
        from matplotlib.colors import Normalize
        from matplotlib import pyplot as plt
        import copy

        plot_kwargs_copy = copy.deepcopy(plot_kwargs)
        if 'norm' not in plot_kwargs_copy:
            plot_kwargs_copy['norm'] = 'log'


        if 'colormap' not in plot_kwargs_copy:
            plot_kwargs_copy['colormap'] = 'Blues'

        if ax is None:
            fig, ax = plt.subplots(1,1)

        five_true_energy_values = np.array(pick_5_values(self.true_binning_geometry.energy_axis.value))*self.true_binning_geometry.energy_axis.unit

        centre_spatial = self.true_binning_geometry.spatial_centre

        # Select the 'Blues' colormap
        cmap = cm.get_cmap(plot_kwargs['colormap'])

        # Normalize the line indices to [0, 1] for the colormap
        norm = Normalize(vmin=-3, vmax=4)



        for _energy_idx, energy_value in enumerate(five_true_energy_values):

            log_edisp_vals = self.log_edisp(recon_energy=self.binning_geometry.energy_axis, 
                                                          true_energy=energy_value,
                                                          true_lon=centre_spatial[0],
                                                          true_lat=centre_spatial[1])
            
            log_edisp_norm = self.logspace_integrator(log_edisp_vals, axes=[self.binning_geometry.energy_axis.value,])
            
            color = cmap(norm(_energy_idx)) 

            if not np.isinf(log_edisp_norm):
                log_edisp_vals = log_edisp_vals - log_edisp_norm
            ax.plot(self.binning_geometry.energy_axis.value, np.exp(log_edisp_vals), color=color)
            ax.axvline(energy_value.value, ls='--', color=color)

            ax.set_xscale('log')

        legend=ax.legend(title="slice at ("+f"{centre_spatial[0].value:.1f}, {centre_spatial[1].value:.1f}"+") +"+f"{centre_spatial.unit.to_string('latex')}")
        plt.setp(legend.get_title(),fontsize='8')

        ax.set_xlabel(r"Energy ["+self.binning_geometry.energy_axis.unit.to_string('latex')+"]")

        ax.set_ylabel(r"Probability Density [1/"+self.binning_geometry.energy_axis.unit.to_string('latex')+"]")
        ax.set_yscale(plot_kwargs['norm'])

        ax.text(0.1, 1.05, r'E$_{\rm{disp}}$', transform=ax.transAxes,
                fontsize=10, ha='center', va='center', fontweight='bold')

        ymin, ymax = ax.get_ylim()

        if ymin < 1e-5:
            ax.set_ylim(bottom=1e-5, top=ymax)
            
        return ax
    

    def _plot_psf_pdf(self, ax, pcolormesh_kwargs={}):

        from matplotlib import cm
        from matplotlib.colors import Normalize
        from matplotlib import pyplot as plt
        import copy

        if ax is None:
            fig, ax = plt.subplots(1,1)

        pcolormesh_kwargs_copy = copy.deepcopy(pcolormesh_kwargs)
        pcolormesh_kwargs_copy.setdefault('vmin', 1e-5)

        if 'norm' not in pcolormesh_kwargs_copy:
            pcolormesh_kwargs_copy['norm'] = 'log'

        if 'cmap' not in pcolormesh_kwargs_copy:
            pcolormesh_kwargs_copy['cmap'] = 'Blues'



        lon_unit = self.binning_geometry.lon_axis.unit
        lat_unit = self.binning_geometry.lat_axis.unit




        centre_spatial = self.true_binning_geometry.spatial_centre


        zoom_lon_slice = np.where(0.5-np.abs(self.binning_geometry.lon_axis.value-centre_spatial[0].value)>=0)
        zoom_lat_slice = np.where(0.5-np.abs(self.binning_geometry.lat_axis.value-centre_spatial[1].value)>=0)


        lon_mesh, lat_mesh = np.meshgrid(self.binning_geometry.lon_axis[zoom_lon_slice], 
                                         self.binning_geometry.lat_axis[zoom_lat_slice], indexing='ij')


        log_psf_values = self.log_psf(recon_lon=lon_mesh.flatten(),
                                  recon_lat=lat_mesh.flatten(), 
                                  true_energy=1*u.TeV,
                                  true_lon=centre_spatial[0],
                                  true_lat=centre_spatial[1],
                                  ).reshape(lon_mesh.shape)
        

        log_psf_norm = self.logspace_integrator(log_psf_values, 
                                                          axes=[self.binning_geometry.lon_axis[zoom_lon_slice].value, 
                                                                self.binning_geometry.lat_axis[zoom_lat_slice].value,
                                                                ],
                                                          axisindices=(0,1,))
        
        if not np.isinf(log_psf_norm):
            log_psf_values = log_psf_values - log_psf_norm
        
        pcm = ax.pcolormesh(self.binning_geometry.lon_axis.value[zoom_lon_slice], 
                            self.binning_geometry.lat_axis.value[zoom_lat_slice],
                      np.exp(log_psf_values).T, **pcolormesh_kwargs_copy)
        
        ax.set_xlabel(r"Longitude ["+(lon_unit).to_string('latex')+']')
        ax.set_ylabel(r"Latitude ["+(lat_unit).to_string('latex')+']')
        ax.text(0.1, 1.05, 'PSF', transform=ax.transAxes,
                fontsize=10, ha='center', va='center', fontweight='bold')
        
        plt.colorbar(mappable=pcm, label=r'Probability Density [1/'+(lon_unit*lat_unit).to_string('latex')+']', ax=ax)

        legend=ax.legend(title=f'''slice at E=1 TeV and 
(l,b) = ({centre_spatial[0].value:.1f}, {centre_spatial[1].value:.1f})''')
        plt.setp(legend.get_title(),fontsize='8')


        ax.set_aspect('equal')

        return ax
        
    def _plot_edisp_density(self, ax, pcolormesh_kwargs={}):

        from matplotlib import cm
        from matplotlib.colors import Normalize
        from matplotlib import pyplot as plt
        import copy

        if ax is None:
            fig, ax = plt.subplots(1,1)

        pcolormesh_kwargs_copy = copy.deepcopy(pcolormesh_kwargs)
        pcolormesh_kwargs_copy.setdefault('vmin', 1e-5)


        energy_unit = self.binning_geometry.energy_axis.unit
        true_energy_unit = self.true_binning_geometry.energy_axis.unit

        meas_energy_mesh, true_energy_mesh = np.meshgrid(self.binning_geometry.energy_axis, 
                                                         self.true_binning_geometry.energy_axis,
                                                         indexing='ij')

        centre_spatial = self.true_binning_geometry.spatial_centre

        log_edisp_vals = self.log_edisp(recon_energy=meas_energy_mesh.flatten(), 
                                        true_energy=true_energy_mesh.flatten(),
                                        true_lon=centre_spatial[0],
                                        true_lat=centre_spatial[1]).reshape(meas_energy_mesh.shape)
        
        log_edisp_norm = self.logspace_integrator(log_edisp_vals, 
                                                            axes=[self.binning_geometry.energy_axis.value,],
                                                            axisindices=[0])
        
        log_edisp_vals = log_edisp_vals - log_edisp_norm[~np.isinf(log_edisp_norm)]
        
        pcm = ax.pcolormesh(self.binning_geometry.energy_axis.value, 
                            self.true_binning_geometry.energy_axis.value,
                            np.exp(log_edisp_vals).T, 
                            **pcolormesh_kwargs_copy)
        
        ax.set_xlabel(r"Energy ["+(energy_unit).to_string('latex')+']')
        ax.set_ylabel(r"True Energy ["+(true_energy_unit).to_string('latex')+']')
        
        plt.colorbar(mappable=pcm, label=r'Probability [1/'+(energy_unit).to_string('latex')+']', ax=ax)

        ax.text(0.1, 1.05, r'E$_{\rm{disp}}$', transform=ax.transAxes,
                fontsize=10, ha='center', va='center', fontweight='bold')


        legend=ax.legend(title=f'''slice at (l,b) = ({centre_spatial[0].value:.1f}, {centre_spatial[1].value:.1f})''')
        plt.setp(legend.get_title(),fontsize='8')

        ax.loglog()
        ax.set_aspect('equal')

        return ax
    

    def _plot_psf_lines(self, ax, plot_kwargs={}):
        from gammabayes.core.utils import pick_5_values
        from matplotlib import cm
        from matplotlib.colors import Normalize
        from matplotlib import pyplot as plt
        import copy

        if ax is None:
            fig, ax = plt.subplots(1,1)


        five_true_energy_values = np.array(pick_5_values(self.true_binning_geometry.energy_axis.value))*self.true_binning_geometry.energy_axis.unit


        lon_unit = self.binning_geometry.lon_axis.unit
        lat_unit = self.binning_geometry.lat_axis.unit




        # Select the 'Blues' colormap
        cmap = cm.get_cmap(plot_kwargs.get('colormap'))

        # Normalize the line indices to [0, 1] for the colormap
        norm = Normalize(vmin=-3, vmax=4)

        true_centre_spatial = self.true_binning_geometry.spatial_centre
        centre_spatial = self.binning_geometry.spatial_centre



        zoom_lon_slice = np.where(0.5-np.abs(self.binning_geometry.lon_axis.value-centre_spatial[0].value)>=0)
        zoom_lat_slice = np.where(0.5-np.abs(self.binning_geometry.lat_axis.value-centre_spatial[1].value)>=0)

        sliced_lon_vals = self.binning_geometry.lon_axis[zoom_lon_slice]
        sliced_lat_vals = self.binning_geometry.lat_axis[zoom_lat_slice]


        centre_lat_slice = np.abs(sliced_lat_vals.value-self.binning_geometry.spatial_centre[0].value).argmin()


        recon_lon_mesh, recon_lat_mesh = np.meshgrid(sliced_lon_vals, 
                                                         sliced_lat_vals,
                                                         indexing='ij')
        
        log_psf_max = -np.inf


        for _energy_idx, energy_value in enumerate(five_true_energy_values):


            log_psf_values = self.log_psf(recon_lon=recon_lon_mesh.flatten(), 
                                        recon_lat=recon_lat_mesh.flatten(),
                                        true_energy=energy_value,
                                        true_lon=true_centre_spatial[0],
                                        true_lat = true_centre_spatial[1]
                                        ).reshape(recon_lon_mesh.shape)
            
            log_psf_norm = self.logspace_integrator(log_psf_values, 
                                                    axes=[sliced_lon_vals.value, 
                                                        sliced_lat_vals.value,
                                                        ],
                                                    axisindices=(0,1,))
        
            log_psf_values = log_psf_values - log_psf_norm

            log_psf_max = max([log_psf_max, np.max(log_psf_values)])

            color = cmap(norm(_energy_idx)) 
            
            ax.plot(sliced_lon_vals, np.exp(log_psf_values[:, centre_lat_slice]).T, color=color, label=f"{energy_value:.1f}")
        

        ax.set_xlabel(r'Longitude ['+(lon_unit).to_string('latex')+']')
        ax.set_ylabel(r'Probability Density [1/'+(lon_unit*lat_unit).to_string('latex')+']')
        ax.set_yscale(plot_kwargs.get('norm'))

        legend=ax.legend(title=f"slice at lat={centre_spatial[0]:.1f}", fontsize=8)
        plt.setp(legend.get_title(),fontsize='8')

        ymin, ymax = ax.get_ylim()

        if ymin < 1e-2:
            ax.set_ylim(bottom=1e-2, ymax=np.exp(log_psf_max+2))

        ax.text(0.1, 1.05, 'PSF', transform=ax.transAxes,
                fontsize=10, ha='center', va='center', fontweight='bold')


        return ax




    def _plot_aeff(self, ax, pcolormesh_kwargs={}, **kwargs):

        from matplotlib import pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1,1)


        energy_unit = self.binning_geometry.energy_axis.unit
        lon_unit = self.binning_geometry.lon_axis.unit

        energy_mesh_values, lon_mesh_values = np.meshgrid(self.true_binning_geometry.energy_axis,
                                                          self.true_binning_geometry.lon_axis,
                                                          indexing='ij')
        
        centre_spatial_values = self.true_binning_geometry.spatial_centre

        log_aeff_vals = self.log_aeff(energy=energy_mesh_values,
                                      lon=lon_mesh_values,
                                      lat=centre_spatial_values[1])


        pcm = ax.pcolormesh(self.true_binning_geometry.energy_axis.value, 
                            self.true_binning_geometry.lon_axis.value,
                            np.exp(log_aeff_vals).T, 
                            **pcolormesh_kwargs)


        plt.colorbar(mappable=pcm, label=r'Effective Area ['+self.aeff_units.to_string('latex')+']', ax=ax)

        ax.text(0.1, 1.05, r'A$_{\rm{eff}}$', transform=ax.transAxes,
                fontsize=10, ha='center', va='center', fontweight='bold')
        
        ax.set_ylabel(r'Longitude ['+(lon_unit).to_string('latex')+']')

        ax.set_xlabel(r"Energy ["+(energy_unit).to_string('latex')+']')
        ax.set_xscale('log')

        legend=ax.legend(title=f'''slice at lat={centre_spatial_values[1].value:.1f})''')
        plt.setp(legend.get_title(),fontsize='8')


        
        return ax
    

    def _plot_ccr(self, ax, plot_kwargs={}):
        from matplotlib import pyplot as plt
        from gammabayes.core.utils import pick_5_values
        from matplotlib import cm
        from matplotlib.colors import Normalize


        if ax is None:
            fig, ax = plt.subplots(1,1)


        energy_unit = self.binning_geometry.energy_axis.unit
        lon_unit = self.binning_geometry.lon_axis.unit

        energy_mesh_values, lon_mesh_values = np.meshgrid(self.true_binning_geometry.energy_axis,
                                                          self.true_binning_geometry.lon_axis,
                                                          indexing='ij')
        
        centre_spatial_values = self.true_binning_geometry.spatial_centre

        log_ccr_vals = self.log_bkg_CCR(energy=energy_mesh_values,
                                      lon=lon_mesh_values,
                                      lat=centre_spatial_values[1])
        
        log_ccr_vals = pick_5_values(log_ccr_vals.T)
        lon_vals = pick_5_values(self.true_binning_geometry.lon_axis)

        # Select the 'Blues' colormap
        cmap = cm.get_cmap(plot_kwargs.get('colormap'))

        # Normalize the line indices to [0, 1] for the colormap
        norm = Normalize(vmin=-3, vmax=4)


        for slice_idx, (ccr_lon_slice, lon_value) in enumerate(zip(log_ccr_vals, lon_vals)):
            color = cmap(norm(slice_idx)) 

            ax.plot(self.true_binning_geometry.energy_axis.value, 
                                np.exp(ccr_lon_slice), 
                                label=f'{lon_value:.1f}', color=color)


        legend = ax.legend(title=f"slice at lat={centre_spatial_values[1]:.1f}", fontsize=8)
        plt.setp(legend.get_title(),fontsize='8')


        ax.text(0.1, 1.05, r'BKG$_{\rm{CCR}}$', transform=ax.transAxes,
                fontsize=10, ha='center', va='center', fontweight='bold')
        
        ax.set_ylabel(r'CCR BKG Rate  ['+self.CCR_BKG_units.to_string('latex')+']',)

        ax.set_xlabel(r"Energy ["+(energy_unit).to_string('latex')+']')
        ax.loglog()

        
        return ax

