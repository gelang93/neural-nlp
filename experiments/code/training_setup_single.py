from train import time_str
from train import ex

model = 'GatedCNNModel'

import sys
file_message = sys.argv[2] if len(sys.argv) > 2 else ""

ex.run(config_updates={'exp_group' : 'single_setup', 'exp_id' : time_str + file_message , 'trainer' : model})