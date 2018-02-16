from train import time_str
from train import ex

model = 'Gated2CNNModel'

import sys
msg = sys.argv[1] if len(sys.argv) > 1 else ''


ex.run(config_updates={'exp_group' : 'beer', 'exp_id' : model + time_str + msg, 'trainer' : model})