from scipy.special import logsumexp
import numpy as np
from gammabayes.samplers import integral_inverse_transform_sampler
from gammabayes.utils import iterate_logspace_integration, construct_log_dx_mesh, update_with_defaults
from gammabayes import EventData, Parameter, ParameterSet


class discrete_loglike(object):
    
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
        """Initialise a discrete_loglikelihood class instance.

        Args:
            logfunction (func): A function that outputs the log 
                likelihood values for the relevant axes and dependent axes. 

            axes (tuple, optional): A tuple of the discrete value axes on which
            the likelihood can/will be evaluated that the likelihood is/would 
            be normalised with respect to. Defaults to None.

            dependent_axes (tuple, optional): A tuple of the discrete value 
            axes on which the likelihood is dependent on that the likelihood is
            not normalised with respect to. Defaults to None.

            name (str, optional): Name given to an instance of the class. 
                Defaults to '[None]'.

            inputunit (list or tuple, optional): A list or tuple containing 
                representations of the units of the 'axes' arguments. 
                Defaults to None.

            axes_names (list, optional): A list of the names of the axes within
            the axes argument. 
            Defaults to ['None'].

            dependent_axes_names (list, optional): A list of the names of the 
            axes within the dependent axesaxes argument. 
            Defaults to ['None'].

            iterative_logspace_integrator (callable, optional): Integration
            method used for normalisation. Defaults to iterate_logspace_integration.
        """
        self.name = name
        self.inputunit = inputunit
        self.logfunction = logfunction
        self.axes_names = axes_names
        self.axes = axes
        print(f'Number of input dimensions {len(self.axes)}')
        self.dependent_axes_names = dependent_axes_names
        self.dependent_axes = dependent_axes
        if len(self.axes)==1 or len(self.dependent_axes)==1:
            if len(self.axes)==1 and len(self.dependent_axes)==1:
                self.axes_dim = 1
                self.dependent_axes_dim = 1
                self.axes_shape = self.axes[0].shape
            elif len(self.axes)==1:
                self.axes_shape = self.axes[0].shape
                self.axes_dim = 1

            else:
                self.axes_dim = len(axes)
                # If it is not done this way self.axes_shape gives out a generator object location instead :(
                self.axes_shape = (*(axis.shape[0] for axis in self.axes),)
                
                self.dependent_axes_dim = 1
        else:
            self.axes_dim = len(axes)
            
            # If it is not done this way self.axes_shape gives out a generator object location instead :(
            self.axes_shape = (*(axis.shape[0] for axis in self.axes),)

            self.dependent_axes_dim = len(self.dependent_axes)

        self.iterative_logspace_integrator  = iterative_logspace_integrator 
        self.parameters               = parameters

        if type(self.parameters)==dict:
            self.parameters = ParameterSet(self.parameters)
        
    def __call__(self, *args, **kwargs) -> np.ndarray | float:
        """Dunder method to be able to use the class in the same method 
        as the logfunction input.
        Args:
            input: Inputs in the same format as would be used for the 
                logfunction used in the class.

        Returns:
            float or np.ndarray: The log-likelihood values for the given inputs.
        """
        update_with_defaults(kwargs, self.parameters)
        return self.logfunction(*args, **kwargs)
    
    
    
    def __repr__(self) -> str:
        """Dunder method for what is the output when `print` is used on a class 
            instance.

        Returns:
            str: A string containing a rough description of the class instance.
        """
        string_text = 'discrete log likelihood class\n'
        string_text = string_text+'-'*(len(string_text)+3)+'\n'
        string_text = string_text+f'name = {self.name}\n'
        string_text = string_text+f'logfunction type is {self.logfunction}\n'
        string_text = string_text+f'input units of {self.inputunit}\n'
        string_text = string_text+f'over axes {self.axes_names}\n'
        string_text = string_text+f'with dependent axes {self.dependent_axes_names}\n'
        
        return string_text
    
    
    
    
    def raw_sample(self, dependentvalues: tuple[float] | list[float] | np.ndarray,  parameters: dict | ParameterSet = {}, numsamples: int = 1) -> np.ndarray:
        """A method to sample the likelihood for given dependent values (e.g. true event data)

        Args:
            dependentvalues (tuple[float] | list[float] | np.ndarray): Dependent values
            to generate a probability distribution.

            numsamples (int, optional): self-explanatory. Defaults to 1.

        Returns:
            np.ndarray: The resultant samples of the given axes.
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
        measured_event_data = EventData(energy=[], glon=[], glat=[], pointing_dirs=[], 
                                        _source_ids=[], obs_id=eventdata.obs_id,
                                        energy_axis=self.axes[0], glongitude_axis=self.axes[1],
                                        glatitude_axis=self.axes[2], 
                                        _true_vals=False)
        for event_datum in eventdata:
            measured_event_data.append(
                self.raw_sample(dependentvalues=event_datum, 
                                parameters=parameters, 
                                numsamples=Nevents_per)
                )

        measured_event_data._source_ids = eventdata._source_ids

        return measured_event_data
    

        
        
        

