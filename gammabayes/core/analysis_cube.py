
from gammabayes import CoordinateGeometry, EventData, Parameter, ParameterSet
from gammabayes.priors import DiscreteLogPrior
from gammabayes.likelihoods import DiscreteLogLikelihood

class AnalysisContainer(object):
    """Class to contain all relevant information required for analysis within GammaBayes."""
    

    def __init__(self, 
                target_priors: list[DiscreteLogPrior],
                observation_likelihoods: DiscreteLogLikelihood | list[DiscreteLogLikelihood], 
                parameter_sets: list[ParameterSet], 
                parameter_info: dict,
                proposal_priors: list[DiscreteLogPrior],
                analysis_method: str = 'sample', 
                data_geometry: CoordinateGeometry = None,
                eventdata_container: EventData = None, 
                pixelation_method: str = None):

        self.target_priors = target_priors
        self.observation_likelihoods = observation_likelihoods
        self.parameter_sets = parameter_sets
        self.parameter_info = parameter_info
        self.analysis_method = analysis_method
        self.proposal_priors = proposal_priors
        self.data_geometry = data_geometry
        self.eventdata_container = eventdata_container
        self.pixelation_method = pixelation_method


    

