from scipy.special import logsumexp
import numpy as np
from gammabayes.samplers import integral_inverse_transform_sampler
from gammabayes.utils import iterate_logspace_integration, construct_log_dx_mesh
from gammabayes import Parameter, ParameterSet, update_with_defaults, GammaBinning, GammaObs, GammaObsCube
# from gammabayes import EventData
import pickle
from tqdm import tqdm
class DiscreteLogLikelihood(object):
    
    def __init__(self, 
                 logfunction: callable,
                 axes: list[np.ndarray] | tuple[np.ndarray]     = None, 
                 dependent_axes: list[np.ndarray]               = None, 
                 name: list[str] | tuple[str]                   = ['None'], 
                 inputunit: str | list[str] | tuple[str]        = ['None'], 
                 iterative_logspace_integrator: callable        = iterate_logspace_integration,
                 parameters: dict | ParameterSet          = ParameterSet(),
                 binning_geometry: GammaBinning = None,
                 true_binning_geometry: GammaBinning = None
                 ) -> None:
        """
        Initializes a DiscreteLogLikelihood object that computes log likelihood values
        for given sets of independent (axes) and dependent (dependent_axes) variables.

        Parameters:
            logfunction (callable): Function that computes log likelihood values.
            
            axes (list[np.ndarray] | tuple[np.ndarray]): Arrays defining the independent variable axes.
            
            dependent_axes (list[np.ndarray]): Arrays defining the dependent variable axes.
            
            name (list[str] | tuple[str], optional): Identifier name(s) for the likelihood instance.
            
            inputunit (str | list[str] | tuple[str], optional): Unit(s) of the input axes.
            
            axes_names (list[str] | tuple[str], optional): Names of the independent variable axes.
            
            dependent_axes_names (list[str] | tuple[str], optional): Names of the dependent variable axes.
            
            iterative_logspace_integrator (callable, optional): Integration method used for normalization.
            
            parameters (dict | ParameterSet, optional): Parameters for the log likelihood function.

        This class facilitates the calculation of log likelihood over discrete spaces, suitable for
        models where likelihood evaluations are essential for parameter inference.
        """

        self.name = name
        self.inputunit = inputunit
        self.logfunction = logfunction
        self.axes = axes
        self.dependent_axes = dependent_axes


        self._create_geometry(axes=axes, dependent_axes=dependent_axes, binning=binning_geometry, true_binning=true_binning_geometry)

        # TODO: Delegate to binning geometries
        self.axes_dim, self.axes_shape, self.dependent_axes_dim = self._get_axis_dims(self.axes, self.dependent_axes)

        self.logspace_integrator  = iterative_logspace_integrator 
        self.parameters               = parameters

        self.parameters = ParameterSet(self.parameters)
    

    def _create_geometry(self, axes, dependent_axes, binning, true_binning):


        if not(axes is None):
            self.binning_geometry = GammaBinning(energy_axis=axes[0], lon_axis=axes[1], lat_axis=axes[2])
        elif not(binning is None):
            self.binning_geometry = binning
        else:
            self.binning_geometry = None


        if not(dependent_axes is None):
            self.true_binning_geometry = GammaBinning(energy_axis=dependent_axes[0], lon_axis=dependent_axes[1], lat_axis=dependent_axes[2])
        elif not(binning is None):
            self.true_binning_geometry = true_binning
        else:
            self.true_binning_geometry = None

        self.axes = self.binning_geometry.axes
        self.dependent_axes = self.true_binning_geometry.axes
            


    def _get_axis_dims(self, axes, dependent_axes):
        """
        Determines the dimensions and shapes of the axes and dependent_axes.

        Parameters:
            axes: Independent variable axes.
            dependent_axes: Dependent variable axes.

        Returns:
            A tuple containing axes dimensions, axes shapes, and dependent axes dimensions.
        """

        if len(axes)==1 or len(dependent_axes)==1:
            if len(axes)==1 and len(dependent_axes)==1:
                axes_dim = 1
                dependent_axes_dim = 1
                axes_shape =  axes[0].shape
            elif len(axes)==1:
                axes_shape = axes[0].shape
                axes_dim = 1

            else:
                axes_dim = len(axes)
                # If it is not done this way axes_shape gives out a generator object location instead :(
                axes_shape = (*(axis.shape[0] for axis in axes),)
                
                dependent_axes_dim = 1
        else:
            axes_dim = len(axes)
            
            # If it is not done this way axes_shape gives out a generator object location instead :(
            axes_shape = (*(axis.shape[0] for axis in axes),)

            dependent_axes_dim = len(dependent_axes)

        return axes_dim, axes_shape, dependent_axes_dim
    

        
    def __call__(self, *args, **kwargs) -> np.ndarray | float:
        """
        Enables using the DiscreteLogLikelihood instance as a callable, passing
        arguments directly to the logfunction.

        Returns:
            The log likelihood values as computed by the logfunction for the given inputs.
        """
        update_with_defaults(kwargs, self.parameters)
        return self.logfunction(*args, **kwargs)
    
    
    
    def __repr__(self) -> str:
        """
        String representation of the DiscreteLogLikelihood instance, providing a summary.

        Returns:
            A descriptive string of the instance including its configuration.
        """
        string_text = 'discrete log likelihood class\n'
        string_text = string_text+'-'*(len(string_text)+3)+'\n'
        string_text = string_text+f'name = {self.name}\n'
        string_text = string_text+f'logfunction type is {self.logfunction}\n'
        string_text = string_text+f'input units of {self.inputunit}\n'
        
        return string_text
    
    
    
    
    def raw_sample(self, 
                   dependentvalues: tuple[float] | list[float] | np.ndarray,  
                   pointing_dir: np.ndarray = None,
                   parameters: dict | ParameterSet = {}, 
                   numsamples: int = 1) -> GammaObs:
        """
        Samples from the likelihood for given dependent values.

        Parameters:
            dependentvalues: Dependent values for generating a probability distribution.
            parameters (dict | ParameterSet, optional): Parameters for the likelihood function.
            numsamples (int, optional): Number of samples to generate.

        Returns:
            An ndarray of samples from the likelihood distribution.
        """
        update_with_defaults(parameters, self.parameters)

        num_eval_axes = len(self.axes) + len(dependentvalues)

        input_units = []

        for axis in self.axes:
            input_units.append(1)

        for val_idx, dependent_val in enumerate(dependentvalues):
            # input_units.append(self.axes[val_idx].unit)
            input_units.append(1)

        for parameter_val in parameters.values():
            input_units.append(1)



        inputmesh = np.meshgrid(*self.axes, *dependentvalues, *parameters.values(), indexing='ij')        

        flattened_meshes = [inputaxis.flatten()*unit for inputaxis, unit in zip(inputmesh, input_units)]
        
        loglikevals = np.squeeze(
            self(
                *flattened_meshes[:num_eval_axes], 
                parameters={key:flattened_meshes[num_eval_axes+param_idx] for param_idx, key in enumerate(parameters.keys())}
                ).reshape(inputmesh[0].shape))

        loglikevals = loglikevals - self.logspace_integrator(loglikevals, axes=[axis.value for axis in self.axes])


        # Used for pseudo-riemann summing
        logdx = construct_log_dx_mesh(self.axes)

        simvals = integral_inverse_transform_sampler(loglikevals+logdx, axes=self.axes, Nsamples=numsamples)
            
        return GammaObs(energy=simvals[0], 
                        lon=simvals[1], 
                        lat=simvals[2], 
                        binning_geometry=self.binning_geometry,
                        irf_loglike=self,
                        pointing_dir=pointing_dir
                        )
    

    def sample(self,eventdata: GammaObs, parameters: dict | ParameterSet = {}, print_progress=False):
        """
        Generates samples from the likelihood based on observed event data.

        Parameters:
            eventdata (EventData): Observed event data to base the sampling on.

            parameters (dict | ParameterSet, optional): Parameters for the likelihood function.

            Nevents_per (int, optional): Number of measured events to sample per true event. 
            (unless you really want to, leave it at 1)

        Returns:
            An EventData instance containing the sampled data.
        """

        measured_event_data = GammaObs(energy=[], lon=[], lat=[], pointing_dir=eventdata.pointing_dir, 
                                        binning_geometry=self.binning_geometry, irf_loglike=self)
        
        data_to_iterate_over = zip(*eventdata.nonzero_bin_data)

        if print_progress:
            data_to_iterate_over = tqdm(data_to_iterate_over, total=len(eventdata.nonzero_bin_data[0]))


        for datum_coord, num_datum in data_to_iterate_over:
            measured_event_data+=self.raw_sample(
                dependentvalues=datum_coord, 
                parameters=parameters, 
                pointing_dir=eventdata.pointing_dir,
                numsamples=num_datum)

        return measured_event_data
    
    

    def save(self, file_name:str , write_mode:str = 'wb'):
        """
        Saves the DiscreteLogLikelihood data to an HDF5 file.

        Args:
        file_name (str): The name of the file to save the data to.
        """

        if not(file_name.endswith('.pkl')):
            file_name = file_name+'.pkl'

        pickle.dump(self, open(file_name,write_mode))

    @classmethod
    def load(cls, file_name:str, read_mode:str = 'rb'):
        """
        Loads the DiscreteLogLikelihood data from a pickle file.

        Parameters:
            file_name (str): The name of the file to load the data from.

        Returns:
            DiscreteLogLikelihood: An instance of the class with the loaded data.
        """

        return  pickle.load(open(file_name,read_mode))