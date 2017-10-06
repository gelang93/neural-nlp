from trainers import *
from sacred import Experiment
import time
time_str = time.ctime().replace(' ', '')

ex = Experiment()


@ex.config
def my_config():
    batch_size = 128
    n_folds = 5
    fold = 0
    optimizer = 'adam'
    metric = 'loss'
    callbacks = 'ss,st,cb,fl,es,cv'
    trainer = 'SharedCNNSiameseTrainer'
    loss = 'hinge'
    nb_train = 1.
    log_full = 'False'
    train_size = .90
    inputs = ['abstract', 'population', 'intervention', 'outcome']
    exp_group = 'population_' + time_str
    exp_id = 0
    nb_epoch = 20
    aspect = 'population'
    pico_file = '../data/files/study_inclusion.csv'


@ex.automain
def main(_config, _run):
    """Run a sacred experiment

    Parameters
    ----------
    _config : special dict populated by sacred with the local variables computed
    in my_config() which can be overridden from the command line or with
    ex.run(config_updates=<dict containing config values>)
    _run : special object passed in by sacred which contains (among other
    things) the name of this run

    This function will be run if this script is run either from the command line
    with

    $ python train.py

    or from within python by

    >>> from train import ex
    >>> ex.run()

    """
    _config['name'] = _run.meta_info['options']['--name']

    trainer = eval(_config['trainer'])(_config)
    trainer.load_cohen_data()
    trainer.load_data_all_fields()
    trainer.common_build_model()
    trainer.build_model()
    trainer.compile_model()
    result = trainer.fit()

    return result
