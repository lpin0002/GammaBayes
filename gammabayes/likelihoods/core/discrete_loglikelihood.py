from scipy.special import logsumexp
import numpy as np
from gammabayes.samplers import integral_inverse_transform_sampler
from gammabayes.utils import iterate_logspace_integration, construct_log_dx_mesh, update_with_defaults
from gammabayes import EventData, Parameter, ParameterSet
import pickle

class DiscreteLogLikelihood(object):
    
    def __init__(self, 
                 logfunction: callable,
                 axes: list[np.ndarray] | tuple[np.ndarray], 
                 dependent_axes: list[np.ndarray], 
                 name: list[str] | tuple[str]                   = ['None'], 
                 inputunit: str | list[str] | tuple[str]        = ['None'], 
                 axes_names: list[str] | tuple[str]             = ['None'], 
                 dependent_axes_names: list[str] | tuple[str]   = ['None'], 
                 iterative_logspace_integrator: callable        = iterate_logspace_integration,
                 parameters: dict | ParameterSet          = ParameterSet(),
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
        self.axes_names = axes_names
        self.axes = axes
        print(f'Number of input dimensions {len(self.axes)}')
        self.dependent_axes_names = dependent_axes_names
        self.dependent_axes = dependent_axes

        self.axes_dim, self.axes_shape, self.dependent_axes_dim = self._get_axis_dims(self.axes, self.dependent_axes)

        self.iterative_logspace_integrator  = iterative_logspace_integrator 
        self.parameters               = parameters

        self.parameters = ParameterSet(self.parameters)


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
        string_text = string_text+f'over axes {self.axes_names}\n'
        string_text = string_text+f'with dependent axes {self.dependent_axes_names}\n'
        
        return string_text
    
    
    
    
    def raw_sample(self, 
                   dependentvalues: tuple[float] | list[float] | np.ndarray,  
                   parameters: dict | ParameterSet = {}, 
                   numsamples: int = 1) -> np.ndarray:
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

        num_non_axes = len(self.axes) + len(dependentvalues)


        inputmesh = np.meshgrid(*self.axes, *dependentvalues, *parameters.values(), indexing='ij')        

        flattened_meshes = [inputaxis.flatten() for inputaxis in inputmesh]
        
        loglikevals = np.squeeze(
            self(
                *flattened_meshes[:num_non_axes], 
                parameters={key:flattened_meshes[num_non_axes+param_idx] for param_idx, key in enumerate(parameters.keys())}
                ).reshape(inputmesh[0].shape))

        loglikevals = loglikevals - self.iterative_logspace_integrator(loglikevals, axes=self.axes)


        # Used for pseudo-riemann summing
        logdx = construct_log_dx_mesh(self.axes)

        simvals = integral_inverse_transform_sampler(loglikevals+logdx, axes=self.axes, Nsamples=numsamples)
            
        return EventData(data=np.asarray(simvals).T, 
                             energy_axis=self.axes[0], 
                             glongitude_axis=self.axes[1], 
                             glatitude_axis=self.axes[2], 
                             _likelihood_id=self.name,
                             _true_vals = False
                             )
    

    def sample(self,eventdata: EventData, parameters: dict | ParameterSet = {}, Nevents_per: int =1):
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

        if hasattr(eventdata, 'obs_id'):
            obs_id = eventdata.obs_id
        else:
            obs_id = 'NoID'

        measured_event_data = EventData(energy=[], glon=[], glat=[], pointing_dirs=[], 
                                        _source_ids=[], obs_id=obs_id,
                                        energy_axis=self.axes[0], glongitude_axis=self.axes[1],
                                        glatitude_axis=self.axes[2], 
                                        _true_vals=False)
        for event_datum in eventdata:
            measured_event_data.append(
                self.raw_sample(dependentvalues=event_datum, 
                                parameters=parameters, 
                                numsamples=Nevents_per)
                )
        try:
            measured_event_data._source_ids = eventdata._source_ids
        except:
            pass

        return measured_event_data
    
    

    def save(self, file_name:str ):
        """
        Saves the DiscreteLogLikelihood data to an HDF5 file.

        Args:
        file_name (str): The name of the file to save the data to.
        """

        if not(file_name.endswith('.pkl')):
            file_name = file_name+'.pkl'

        pickle.dump(self, open(file_name,'wb'))

    @classmethod
    def load(cls, file_name):
        return  pickle.load(open(file_name,'rb'))