from gammabayes import ParameterSet
import logging, warnings, numpy as np

def _handle_parameter_specification(
        parameter_specifications: dict | ParameterSet,
        num_required_sets: int = None,
        _no_required_num=False):
    """
    Processes and validates parameter specifications against the target priors.

    Parameters:
        parameter_specifications (dict | ParameterSet): Parameter specifications.
        
        log_target_priors (list[ParameterSet] | dict[dict], optional): Target priors.

    Raises:
        ValueError: If the number of hyperparameter axes specified exceeds the number of priors unless
        'self.no_priors_on_init' is True.
    """
    _num_parameter_specifications = len(parameter_specifications)
    formatted_parameter_specifications = []*_num_parameter_specifications

    if _num_parameter_specifications>0:

        if type(parameter_specifications)==dict:

            for single_prior_parameter_specifications in parameter_specifications.items():

                parameter_set = ParameterSet(single_prior_parameter_specifications)

                formatted_parameter_specifications.append(parameter_set)

        elif type(parameter_specifications)==list:
            formatted_parameter_specifications = [
                ParameterSet(
                    single_prior_parameter_specification
                    ) for single_prior_parameter_specification in parameter_specifications
                ]

    if num_required_sets is not None:
        _num_priors = num_required_sets
    else:
        _num_priors = _num_parameter_specifications

    if not _no_required_num or (num_required_sets is not None):

        diff_in_num_hyperaxes_vs_priors = _num_priors-_num_parameter_specifications

        if diff_in_num_hyperaxes_vs_priors<0:
            raise ValueError(f'''
You have specifed {np.abs(diff_in_num_hyperaxes_vs_priors)} more hyperparameter axes than priors.''')
        
        elif diff_in_num_hyperaxes_vs_priors>0:
            warnings.warn(f"""
You have specifed {diff_in_num_hyperaxes_vs_priors} less hyperparameter axes than priors. 
Assigning empty hyperparameter axes for remaining priors.""")
            
            _num_parameter_specifications = len(formatted_parameter_specifications)
            
            for __idx in range(_num_parameter_specifications, _num_priors):
                formatted_parameter_specifications.append(ParameterSet())


    return formatted_parameter_specifications


def _handle_nuisance_axes(nuisance_axes: list[np.ndarray],
                            log_likelihood=None, log_prior=None):
    """
    Handles the assignment and retrieval of nuisance axes. 
    This method first checks if `nuisance_axes` is provided. If not, it attempts to retrieve nuisance axes 
    from `log_likelihood` or `log_prior`. If neither is available, it raises an exception.

    Args:
        nuisance_axes (list[np.ndarray]): A list of numpy arrays representing the nuisance axes.

    Raises:
        Exception: Raised if `nuisance_axes` is not provided and cannot be retrieved from either 
                `log_likelihood` or `log_proposal_prior` OR `log_target_priors`.

    Returns:
        list[np.ndarray]: The list of numpy arrays representing the nuisance axes. This can be either the 
                        provided `nuisance_axes`, or retrieved from `log_likelihood` or `log_priors`.
    """
    if nuisance_axes is None:
        try:
            return log_likelihood.nuisance_axes
        except AttributeError:
            try:
                return log_prior.axes
            except AttributeError:
                raise Exception("Dependent value axes used for calculations not given.")
                
    return nuisance_axes