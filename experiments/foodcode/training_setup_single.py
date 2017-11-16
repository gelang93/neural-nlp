from train import time_str
from train import ex

model = 'SingleCNNModel'

import sys
f = open('../store/message/single_message_' + time_str + '.txt', 'w')
g = open(model + '.py', 'r').read()
f.write(model + '\t' + (sys.argv[1] if len(sys.argv) > 1 else time_str + ' No Message'))
f.write('\n\n' + g)
f.close()


ex.run(config_updates={'exp_group' : 'single_setup', 'exp_id' : time_str, 'trainer' : model})