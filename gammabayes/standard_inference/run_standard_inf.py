import sys, os, warnings

from gammabayes.standard_inference.standard_3comp_bkg import standard_3COMP_BKG
from gammabayes.utils.config_utils import read_config_file

if __name__=="__main__":
    try:
        config_file_path = sys.argv[1]
    except:
        warnings.warn('No configuration file given')
        config_file_path = os.path.dirname(__file__)+'/Z2_DM_3COMP_BKG_config_default.yaml'
    config_dict = read_config_file(config_file_path)
    print(f"initial config_file_path: {config_file_path}")
    standard_3COMP_BKG_instance = standard_3COMP_BKG(config_dict=config_dict,)
    standard_3COMP_BKG_instance.run()
    print("\n\nRun Done. Now saving")
    standard_3COMP_BKG_instance.save()
    print("\n\nResults saved.")