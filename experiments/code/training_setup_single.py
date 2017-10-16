from train import time_str
from train import ex

import sys
f = open('../store/message/message_single_' + time_str + '.txt', 'w')
g = open('trainers.py', 'r').read()
f.write(sys.argv[1] if len(sys.argv) > 1 else time_str + ' No Message')
f.write('\n\n' + g)
f.close()


ex.run(config_updates={'exp_group' : 'single_'+time_str+'_setup'})