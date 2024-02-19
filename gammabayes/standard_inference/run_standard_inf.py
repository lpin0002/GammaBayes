import sys, os, warnings

from gammabayes.standard_inference.standard_3comp_bkg import ScanMarg_ConfigAnalysis
from gammabayes.utils.config_utils import read_config_file, save_config_file

if __name__=="__main__":
    try:
        config_file_path = sys.argv[1]
    except:
        warnings.warn('No configuration file given')
        config_file_path = os.path.dirname(__file__)+'/3COMP_BKG_config_default.yaml'
    config_dict = read_config_file(config_file_path)

    try:
        save_config_file(config_dict, config_dict['save_path']+'config.yaml')
    except:
        pass

    print(f"initial config_file_path: {config_file_path}")
    standard_3COMP_BKG_instance = ScanMarg_ConfigAnalysis(config_dict=config_dict,)
    standard_3COMP_BKG_instance.run()
    print("\n\nRun Done. Now saving")
    standard_3COMP_BKG_instance.discrete_hyper_like_instance.save(config_dict['save_path']+'results.h5')
    print("\n\nResults saved.")


    try:
        
        standard_3COMP_BKG_instance.plot_results(
            **config_dict['plot_results_kwargs'])

    except Exception as excpt:
        print("An error occurred when trying to plot the results:")
        print(excpt)