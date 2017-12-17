from trainers import *
from sacred import Experiment
import time
time_str = time.ctime().replace(' ', '')

ex = Experiment()


@ex.config
def my_config():
    batch_size = 64
    n_folds = 5
    fold = 0
    optimizer = 'adam'
    metric = 'loss'
    callbacks = 'es,cb,fl,cv'
    trainer = 'CountModel'
    loss = 'hinge'
    nb_train = 1.
    log_full = 'False'
    train_size = .90
    exp_group = 'beer'
    exp_id = time_str
    nb_epoch = 10

@ex.automain
def main(_config, _run):
    _config['name'] = _run.meta_info['options']['--name']

    trainer = eval(_config['trainer'])(_config)
    trainer.load_data()
    trainer.build_model()
    trainer.compile_model()
    result = trainer.fit()

    return result