from astropy import units as u
from gammabayes.priors.core.wrappers import _wrap_if_missing_keyword
from gammabayes import update_with_defaults


class BaseSpectral_PriorComp:

    @staticmethod
    def _zero_output(inputval):
        """
        A helper method that returns a zero value for any given input.

        Args:
            inputval: Input value or array.

        Returns:
            The zero value of the same shape as the input.
        """
        return inputval[0]*0
    
    @staticmethod
    def _one_output(inputval):
        """
        A helper method that returns a one value for any given input.

        Args:
            inputval: Input value or array.

        Returns:
            A value or array of ones of the same shape as the input.
        """
        return inputval[0]*0 + 1



    def __init__(self, logfunc, mesh_efficient_logfunc=None, default_parameter_values=None, *args, **kwargs):


        if default_parameter_values is None:
            default_parameter_values = {}
        self.default_parameter_values = default_parameter_values

        self._kwd_arg_string = "kwd_parameters"

        self._input_logfunc = logfunc
        self._input_mesh_efficient_logfunc = mesh_efficient_logfunc

        self.wrapped_logfunc = _wrap_if_missing_keyword(self._input_logfunc, kwarg_string=self._kwd_arg_string, )


    def __call__(self, *args, **kwargs):

        if "kwd_parameters" in kwargs:
            update_with_defaults(kwargs["kwd_parameters"], self.default_parameter_values)
        else:
            update_with_defaults(kwargs, self.default_parameter_values)
            
        return self.wrapped_logfunc(*args, **kwargs)


