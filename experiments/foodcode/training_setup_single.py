from train import time_str
from train import ex

model = 'GatedCNNModel'

import sys
msg = sys.argv[1] if len(sys.argv) > 1 else ''

ex.run(config_updates={'exp_group' : 'food', 'exp_id' : time_str, 'trainer' : model})