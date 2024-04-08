
from gammabayes import DataGeometry, EventData


class AnalysisContainer(object):

    def __init__(self, target_priors, observation_likelihoods, 
                 parameter_info, analysis_method, proposal_priors,
                 data_geometry: DataGeometry = None,
                 eventdata_container: EventData = None, 
                 pixelation_method: str = None):

        self.target_priors = target_priors
        self.observation_likelihoods = observation_likelihoods
        self.parameter_info = parameter_info
        self.analysis_method = analysis_method
        self.proposal_priors = proposal_priors

