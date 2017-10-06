import pickle
import os
from os.path import *

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint, EarlyStopping
import keras.backend as K

from batch_generators import bg1, bg2

from callbacks import *

import sys 
sys.path.insert(0, '../../preprocess')

import vectorizer


class Trainer:
    """Handles the following tasks:

    1. Load embeddings and labels
    2. Build a keras model
    3. Calls fit() to train

    """
    def __init__(self, config):
        """Set attributes

        Attributes
        ----------
        config : sacred config dict

        Also do the train/val split.

        """
        self.C = config

        # load cdnos sorted by the ones with the most studies so we pick those first when undersampling
        df = pd.read_csv(self.C['pico_file'])
        cdnos = df.groupby('cdno').size()
        cdnos = np.array(cdnos[cdnos > 1].sort_values(ascending=False).index)

        # split into train and validation at the cdno-level
        nb_reviews = len(cdnos)
        train_size, nb_train = self.C['train_size'], self.C['nb_train']
        
        train_cdno_idxs, val_cdno_idxs = train_test_split(np.arange(nb_reviews), train_size=train_size, random_state=1337)
        
        first_train = np.floor(len(train_cdno_idxs)*nb_train)
        train_cdno_idxs = np.sort(train_cdno_idxs)[:first_train.astype('int')] # take a subset of the training cdnos
        val_cdno_idxs = np.sort(val_cdno_idxs)
        
        train_cdnos, val_cdnos = set(cdnos[train_cdno_idxs]), set(cdnos[val_cdno_idxs])
        
        train_idxs = np.array(df[df.cdno.isin(train_cdnos)].index)
        val_idxs = np.array(df[df.cdno.isin(val_cdnos)].index)

        self.C['train_idxs'], self.C['val_idxs'] = train_idxs, val_idxs
        self.C['train_cdnos'], self.C['val_cdnos'] = train_cdnos, val_cdnos
        self.C['cdnos'] = df.cdno


    def load_data(self):
        """Load inputs
        
        Parameters
        ----------
        inputs : list of vectorizer names (expected to be in ../data/vectorizers)
        
        """
        self.nb_values = None 
        
        for input in self.C['inputs']:
            self.C[input] = pickle.load(open('../data/vectorizers/{}s.p'.format(input))) 
            if self.nb_values:
                assert self.nb_values == len(self.C[input])
            self.nb_values = len(self.C[input])
            
    def load_data_all_fields(self) :
        self.vec = pickle.load(open('../data/vectorizers/allfields_with_embeddings.p', 'rb'))
        index = self.vec.index
        self.nb_values = None
        for input in self.C['inputs'] :
            input_range = index[input]
            self.C[input] = vectorizer.Vectorizer()
            self.C[input].idx2word = self.vec.idx2word
            self.C[input].word2idx = self.vec.word2idx
            self.C[input].X = self.vec.X[input_range[0]:input_range[1]]
            if self.nb_values: 
                assert self.nb_values == len(self.C[input])
            self.nb_values = len(self.C[input])
            
    def load_cohen_data(self) :
        df = pd.read_csv('../data/files/test_cohen_dedup.csv')
        cdnos = df.groupby('cdno').size()
        cdnos = np.array(cdnos[cdnos > 1].sort_values(ascending=False).index)
        self.C['test_cdnos'] = df.cdno
        self.C['test_idxs'] = np.array(df.index)
        self.C['test_vec'] = pickle.load(open('../data/vectorizers/cohendata_dedup.p', 'rb'))
        

    def common_build_model(self) :
        aspect = self.C['aspect']
        aspect_comp = list(self.C['inputs'])
        aspect_comp.remove('abstract')
        aspect_comp.remove(aspect)

        self.A = [('same_abstract', 'abstract')]
        S = ['same_'+aspect, 'corrupt_'+aspect, 'valid_'+aspect]
        self.S = [(s, aspect) for s in S]
        self.O = [('same_' + s, s) for s in aspect_comp]
        self.field_in_train = zip(*self.A+self.S+self.O)[0]

    def compile_model(self):
        """Compile keras model

        Also define functions for evaluation.

        """
        print 'Compiling...'
        self.model.summary()

    def fit(self):
        """Set up callbacks and start training"""

        # define callbacks
        weight_str = '../store/weights/{}/{}/{}-{}.h5'
        exp_group, exp_id = self.C['exp_group'], self.C['exp_id']
        fold, metric = self.C['fold'], self.C['metric']
        
        weight_str = weight_str.format(exp_group, exp_id, fold, {})
        if not os.path.exists(dirname(weight_str)) :
            os.makedirs(dirname(weight_str))
        
        cb = ModelCheckpoint(weight_str.format(metric),
                             monitor='loss',
                             save_best_only=True,
                             mode='min')
        ce = ModelCheckpoint(weight_str.format('val_loss'),
                             monitor='val_loss', # every time training loss goes down
                             mode='min')
        
        train_idxs, val_idxs = self.C['train_idxs'], self.C['val_idxs']
        
        X = {input: self.C[input].X for input in self.C['inputs']}
        X_train = {input: X_[train_idxs] for input, X_ in X.items()}
        X_val = {input: X_[val_idxs] for input, X_ in X.items()}

        train_cdnos = self.C['cdnos'].iloc[train_idxs].reset_index(drop=True)
        val_cdnos = self.C['cdnos'].iloc[val_idxs].reset_index(drop=True)
        
        batch = bg2(X_val, val_cdnos)
        
        X_test = {'abstract': self.C['test_vec'].X[self.C['test_idxs']]}
        test_cdnos = self.C['test_cdnos'].reset_index(drop=True)
        batch_test = bg2(X_test, test_cdnos, 500)
        
        study_dim = self.model.get_layer('pool').output_shape[-1]
        ss = StudySimilarityLogger(next(batch), study_dim)
        sst = StudySimilarityLogger(next(batch_test), study_dim, logname='test_similarity')
        # pl = PrecisionLogger(X_val, study_dim=self.model.get_layer('study').output_shape[-1])

        es = EarlyStopping(monitor='val_similarity', patience=10, verbose=2, mode='max')
        fl = Flusher()
        cv = CSVLogger(exp_group, exp_id, fold)
        # sl = StudyLogger(X_study[val_idxs], self.exp_group, self.exp_id)
        idx2word = self.C['abstract'].idx2word
        lw = LargeWordCallback(idx2word)

        # filter down callbacks
        callback_dict = {'cb': cb, # checkpoint best
                         'ce': ce, # checkpoint every
                         'fl': fl, # flusher
                         'es': es, # early stopping
                         # 'sl': sl, # study logger
                         # 'pl': pl, # precision logger
                         'ss': ss, # study similarity logger
                         'st': sst,
                         'cv': cv, # should go *last* as other callbacks populate `logs` dict
                         #'lw': lw, # large word callback
        }
        callback_list = self.C['callbacks'].split(',')
        self.callbacks = [callback_dict[cb_name] for cb_name in callback_list]
        
        gen_source_target_batches = bg1(X_train, train_cdnos, self)
        batch_size, nb_epoch = self.C['batch_size'], self.C['nb_epoch']

        nb_train = len(train_idxs)
        self.model.fit_generator(gen_source_target_batches,
                                 steps_per_epoch=(nb_train/batch_size),
                                 epochs=nb_epoch,
                                 verbose=2,
                                 callbacks=self.callbacks)
