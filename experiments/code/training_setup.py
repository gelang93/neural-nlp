from train import time_str
from train import ex

fields = ['intervention', 'population', 'outcome']

for f in fields :
    ex.run(config_updates={'exp_group' : f+'_'+time_str, 'aspect' : f})