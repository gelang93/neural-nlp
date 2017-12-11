from trainers import *
from sacred import Experiment
import time
time_str = time.ctime().replace(' ', '')

ex = Experiment()


@ex.config
def my_config():
    batch_size = 32
    n_folds = 5
    fold = 0
    optimizer = 'adam'
    metric = 'loss'
    callbacks = 'ss,st,al,cb,fl,es,cv'
    trainer = 'SeparateCNNModel'
    loss = 'hinge'
    nb_train = 1.
    log_full = 'False'
    train_size = .90
    inputs = ['abstract', 'population', 'intervention', 'outcome']
    exp_group = 'population'
    exp_id = time_str
    nb_epoch = 20
    aspect = 'population'
    pico_file = '../data/files/study_inclusion.csv'


@ex.automain
def main(_config, _run):
    _config['name'] = _run.meta_info['options']['--name']

    trainer = eval(_config['trainer'])(_config)
    trainer.load_data_all_fields()
    trainer.load_cohen_data()

    trainer.common_build_model()
    trainer.build_model()
    trainer.compile_model()
    result = trainer.fit()

    return result
