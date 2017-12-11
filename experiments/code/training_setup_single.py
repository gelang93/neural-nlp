from train import time_str
from train import ex

model = 'GatedCNNModel'

import sys
file_message = sys.argv[2] if len(sys.argv) > 2 else ""
f = open('../store/message/single_message_' + file_message + time_str + '.txt', 'w')
g = open(model + '.py', 'r').read()

f.write(model + '\t' + (sys.argv[1] if len(sys.argv) > 1 else time_str + ' No Message'))
f.write('\n\n' + g)
f.close()



ex.run(config_updates={'exp_group' : 'single_setup', 'exp_id' : time_str + file_message , 'trainer' : model})