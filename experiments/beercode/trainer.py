import pickle
import os
from os.path import *

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import keras.backend as K

from batch_generators import bg1

from callbacks import *

import sys 
sys.path.insert(0, '../../preprocess')

import vectorizer


class Trainer:
    def __init__(self, config):
        self.C = config
            
    def load_data(self) :
        self.vec = pickle.load(open('../../beer_data/beer_vec_ds_df10.p', 'rb'))
        # print self.vec.vocab_size
        # X_tf = np.zeros((self.vec.X.shape[0], self.vec.vocab_size))
        # for i in range(len(self.vec.X)) :
        #     X_tf[i, self.vec.X[i, :]] = 1.

        # X_tf = X_tf[:, 2:]
        # self.vec.X = X_tf

        ds = pd.read_csv('../../beer_data/beer_ds.csv')
        
        from ast import literal_eval as make_tuple
        ds['bits'] = ds['bits'].map(lambda s : make_tuple(s))
            
        aspect_columns = sorted(['review/appearance', 'review/taste', 'review/aroma', 'review/palate'])
        
        train_idxs, val_idxs = train_test_split(ds.index, stratify=ds['bits'], train_size=0.9, random_state=1337)
        self.C['train_idxs'], self.C['val_idxs'] = train_idxs, val_idxs

        print len(train_idxs), len(val_idxs)

        self.ds = ds
        self.aspect_columns = aspect_columns
        self.modifier = ['D', 'S']

    def compile_model(self):
        print 'Compiling...'
        self.model.summary()

    def fit(self):
        weight_str = '../store/weights/{}/{}/{}.h5'
        exp_group, exp_id = self.C['exp_group'], self.C['exp_id']
        metric = self.C['metric']
        
        weight_str = weight_str.format(exp_group, exp_id, {})
        if not os.path.exists(dirname(weight_str)) :
            os.makedirs(dirname(weight_str))
        
        cb = ModelCheckpoint(weight_str.format(metric),
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min')
        
        train_idxs, val_idxs = self.C['train_idxs'], self.C['val_idxs']
        batch_size, nb_epoch = self.C['batch_size'], self.C['nb_epoch']

        
        train_gen = bg1(self.vec.X, self.ds.loc[train_idxs], self, nb_sample=batch_size)
        val_gen = bg1(self.vec.X, self.ds.loc[val_idxs], self, nb_sample=batch_size)

        pal = PerAspectAUCLogger(X=self.vec.X, idxs=val_idxs, trainer=self, batch_size=batch_size)
        es = EarlyStopping(monitor='val_loss', patience=2, verbose=2, mode='min')
        fl = Flusher()
        cv = CSVLogger(exp_group, exp_id)
        
        tb = TensorBoard()
        
        callback_dict = {#'al': pal,
                         'cb': cb, # checkpoint best
                         'fl': fl, # flusher
                         'es': es, # early stopping
                         'cv': cv, # should go *last* as other callbacks populate `logs` dict
        }
        
        callback_list = self.C['callbacks'].split(',')
        self.callbacks = [callback_dict[cb_name] for cb_name in callback_list]+[tb]
        
        nb_train = len(train_idxs)
        self.model.fit_generator(train_gen,
                                 steps_per_epoch=(nb_train/batch_size),
                                 epochs=nb_epoch,
                                 verbose=1,
                                 validation_data=val_gen,
                                 validation_steps=2,
                                 callbacks=self.callbacks)