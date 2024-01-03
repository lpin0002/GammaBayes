
from gammabayes.utils.config_utils import read_config_file
from gammabayes.utils.ozstar.make_ozstar_scripts import makejobscripts
import sys, os
from warnings import warn


try:
    config_file_path = sys.argv[1]
    config_inputs = read_config_file(config_file_path)


except KeyboardInterrupt:
    raise KeyboardInterrupt

except:
    config_file_path = os.path.dirname(__file__)+'/default_ozstar_config.yaml'
    print(config_file_path)
    config_inputs = read_config_file(config_file_path)
    print(config_inputs)

    print(makejobscripts(config_inputs, config_file_path, path_to_run_file='gammabayes.standard_inference.Z3_DM_3COMP_BKG'))
