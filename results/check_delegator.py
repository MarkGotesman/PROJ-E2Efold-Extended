# Used to delegate (meaning, input to) the ```check_*.py``` modules in this folder.
import _pickle as pickle
from e2efold.common.config import process_config

experiment_archiveii_config        = process_config ('../experiment_archiveii/config.json')
experiment_archiveii_config_long   = process_config ('../experiment_archiveii/config_long.json')
experiment_rnastralign_config      = process_config ('../experiment_rnastralign/config.json')
experiment_rnastralign_config_long = process_config ('../experiment_rnastralign/config_long.json')

# with open('EXPER-ArchiveII-Short_prediction_dict.pickle', 'rb') as f:
# 	ct_dict = pickle.load(f)

# print(ct_dict)