import sys, os, warnings, numpy as np

from gammabayes.utils.config_utils import read_config_file, save_config_file

if __name__=="__main__":
    try:
        config_file_path = sys.argv[1]
    except:
        warnings.warn('No configuration file given')
        config_file_path = os.path.dirname(__file__)+'/3COMP_BKG_config_default.yaml'
    config_dict = read_config_file(config_file_path)

    if 'path_to_measured_event_data' in config_dict:
        path_to_measured_event_data = config_dict['path_to_measured_event_data']
    else:
        path_to_measured_event_data = None


    if 'save_path_for_measured_event_data' in config_dict:
        save_path_for_measured_event_data = config_dict['save_path_for_measured_event_data']
    else:
        save_path_for_measured_event_data = config_dict['save_path']+'event_data.h5'

    config_dict['save_path_for_measured_event_data'] = save_path_for_measured_event_data

    try:
        save_config_file(config_dict, config_dict['save_path']+'config.yaml')
    except:
        pass

    # A bit nasty but it works
    if 'num_bkg_comp' in config_dict:
        if config_dict['num_bkg_comp']==1:
            from gammabayes.high_level_inference.standard_1comp_bkg import ScanMarg_ConfigAnalysis
        if config_dict['num_bkg_comp']==3:
            from gammabayes.high_level_inference.standard_3comp_bkg import ScanMarg_ConfigAnalysis
    else:
        from gammabayes.high_level_inference.standard_3comp_bkg import ScanMarg_ConfigAnalysis


    print(f"initial config_file_path: {config_file_path}")
    high_level_inference_instance = ScanMarg_ConfigAnalysis(config_dict=config_dict,)
    high_level_inference_instance.run(path_to_measured_event_data=path_to_measured_event_data)
    print("\n\nRun Done. Now saving")


    if 'result_save_filename' in config_dict:
        result_save_filename = config_dict['result_save_filename']
    else:
        result_save_filename = 'results.h5'

    high_level_inference_instance.discrete_hyper_like_instance.save(config_dict['save_path']+result_save_filename)
    print("\n\nResults saved.")


    # try:
        
    #     standard_3COMP_BKG_instance.plot_results(
    #         **config_dict['plot_results_kwargs'])

    # except Exception as excpt:
    #     print("An error occurred when trying to plot the results:")
    #     print(excpt)