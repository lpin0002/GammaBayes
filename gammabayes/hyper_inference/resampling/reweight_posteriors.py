
import dynesty
import numpy as np

def reweight_samples(sampler_results, original_parameter_collection, target_parameter_collection, mixture_tree=None):







    if not(mixture_tree is None):

        # Making sure that the order of the mixture parameters is the same as the tree layout
        mixture_param_order = mixture_tree.nodes.copy()
        del mixture_param_order['root']
        original_parameter_collection.reorder(list(mixture_param_order))


        # Presuming that the layout of the mixture tree is the same
        target_parameter_collection.reorder(list(mixture_param_order))


    original_parameter_collection_logpdf = original_parameter_collection.logpdf
    target_parameter_collection_logpdf = target_parameter_collection.logpdf


    # Calculate the weights for each sample
    proposal_weights = np.array([original_parameter_collection_logpdf(theta) for theta in sampler_results.samples])
    target_weights = np.array([target_parameter_collection_logpdf(theta) for theta in sampler_results.samples])


    # # Use dynesty.utils.resample_equal to reweight samples
    resampled_results = dynesty.utils.reweight_run(sampler_results, logp_new=target_weights, logp_old=proposal_weights)


    return resampled_results