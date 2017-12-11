from train import time_str
from train import ex

model = 'Gated2CNNModel'

import sys
msg = sys.argv[2] if len(sys.argv) > 2 else ''
f = open('../store/message_beer/' + msg + model + '_' + time_str + '.py', 'w')
g = open(model + '.py', 'r').read()
f.write('#' + model + '\t' + (sys.argv[1] if len(sys.argv) > 1 else time_str + ' No Message'))
f.write('\n\n' + g)
f.close()


ex.run(config_updates={'exp_group' : 'beer', 'exp_id' : model + time_str + msg, 'trainer' : model})