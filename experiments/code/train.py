import os
import sys

import time
import plac
import pickle

import numpy as np
import pandas as pd

from sklearn.cross_validation import KFold

from trainers import CNNTrainer


@plac.annotations(
        exp_group=('the name of the experiment group for loading weights', 'option', None, str),
        exp_id=('id of the experiment - usually an integer', 'option', None, str),
        nb_epoch=('number of epochs', 'option', None, int),
        nb_filter=('number of filters', 'option', None, int),
        filter_lens=('length of filters', 'option', None, str),
        nb_hidden=('number of hidden states', 'option', None, int),
        hidden_dim=('size of hidden state', 'option', None, int),
        dropout_prob=('dropout probability', 'option', None, float),
        dropout_emb=('perform dropout after the embedding layer', 'option', None, str),
        reg=('l2 regularization constant', 'option', None, float),
        backprop_emb=('whether to backprop into embeddings', 'option', None, str),
        batch_size=('batch size', 'option', None, int),
        word2vec_init=('initialize embeddings with word2vec', 'option', None, str),
        nb_train=('number of examples to train on', 'option', None, int),
        nb_val=('number of examples to validate on', 'option', None, int),
        n_folds=('number of folds for cross validation', 'option', None, int),
        optimizer=('optimizer to use during training', 'option', None, str),
        lr=('learning rate to use during training', 'option', None, float),
        do_cv=('do cross validation if true', 'option', None, str),
        metric=('metric to use during training (acc or f1)', 'option', None, str),
        callbacks=('list callbacks to use during training', 'option', None, str),
        trainer=('type of trainer to use', 'option', None, str),
        features=('list of additional features to use', 'option', None, str),
        inputs=('data to use for input', 'option', None, str),
        labels=('labels to use', 'option', None, str),
)
def main(exp_group='', exp_id='', nb_epoch=5, nb_filter=1000, filter_lens='1,2,3', 
        nb_hidden=1, hidden_dim=1024, dropout_prob=.5, dropout_emb='True', reg=0,
        backprop_emb='False', batch_size=128, word2vec_init='False', nb_train=1000000,
        nb_val=1000000, n_folds=5, optimizer='adam', lr=.001,
        do_cv='False', metric='val_main_acc', callbacks='cb,ce,fl,cv,es',
        trainer='CNNTrainer', features='', inputs='abstracts', labels='outcomes'):
    """Training process

    1. Parse command line arguments
    2. Load input data and labels
    3. Build the keras model
    4. Train the model
    5. Log training information

    """
    # collect hyperparams for visualization code
    args = sys.argv[1:]
    pnames, pvalues = [pname.lstrip('-') for pname in args[::2]], args[1::2]
    hyperparam_dict = {pname: pvalue for pname, pvalue in zip(pnames, pvalues)}

    # parse command line options
    filter_lens = [int(filter_len) for filter_len in filter_lens.split(',')]
    backprop_emb = True if backprop_emb == 'True' else False
    dropout_emb = dropout_prob if dropout_emb == 'True' else 1e-100
    word2vec_init = True if word2vec_init == 'True' else False
    do_cv = True if do_cv == 'True' else False
    nb_filter /= len(filter_lens) # make it so there are only nb_filter *total* - NOT nb_filter*len(filter_lens)
    callbacks = callbacks.split(',')
    features = features.split(',') if features != '' else []
    inputs = inputs.split(',')

    # load data and supervision
    trainer = eval(trainer)(exp_group, exp_id, hyperparam_dict, trainer)
    trainer.load_texts(inputs)
    trainer.load_auxiliary(features)
    trainer.load_labels(labels)

    # set up fold(s)
    nb_texts = len(trainer.X_vecs[0].X)
    folds = KFold(nb_texts, n_folds, shuffle=True, random_state=1337) # for reproducibility!
    if not do_cv:
        folds = list(folds)[:1] # only do the first fold if not doing cross-valiadtion

    # cross-fold training
    for fold_idx, (train_idxs, val_idxs) in enumerate(folds):
        # model
        trainer.build_model(nb_filter, filter_lens, nb_hidden, hidden_dim,
                dropout_prob, dropout_emb, backprop_emb, word2vec_init)
        trainer.compile_model(metric, optimizer, lr)
        trainer.save_architecture()

        # train
        history = trainer.train(train_idxs, val_idxs, nb_epoch, batch_size,
                nb_train, nb_val, callbacks, fold_idx, metric)


if __name__ == '__main__':
    plac.call(main)