import pickle
import os
from os.path import *

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
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

            
    def load_data_all_fields(self) :
        self.vec = pickle.load(open('../data/vectorizers/allfields_with_embedding_5000.p', 'rb'))
        index = self.vec.index
        self.nb_values = None
        for input in self.C['inputs'] :
            input_range = index[input]
            self.C[input] = vectorizer.Vectorizer()
            self.C[input].idx2word = self.vec.idx2word
            self.C[input].word2idx = self.vec.word2idx
            self.C[input].X = self.vec.X[input_range[0]:input_range[1]]

            # X_tf = np.zeros((self.C[input].X.shape[0], self.vec.vocab_size))
            # for i in range(len(self.C[input].X)) :
            #     X_tf[i, self.C[input].X[i, :]] = 1.

            # X_tf = X_tf[:, 2:]
            # self.C[input].X = X_tf
            # if input in ['population', 'intervention', 'outcome'] :
            #     self.C[input].X = self.C[input].X[:, -200:]
            if self.nb_values: 
                assert self.nb_values == len(self.C[input])
            self.nb_values = len(self.C[input])
            
    def load_cohen_data(self) :
        df = pd.read_csv('../data/files/test_cohen_dedup.csv')
        nb_studies = len(df)
        H = np.zeros((nb_studies, nb_studies))

        cdnos = list(set(df.cdno))
        for i in range(nb_studies) :
            H[i, df[df['cdno'] == df['cdno'][i]].index] = 1
        H[np.arange(nb_studies), np.arange(nb_studies)] = 0
        
        self.C['test_ref'] = H
        self.C['test_cdnos'] = df.cdno
        self.C['test_idxs'] = np.array(df.index)
        self.C['test_vec'] = pickle.load(open('../data/vectorizers/cohendata_dedup_5000.p', 'rb'))

        # X_tf = np.zeros((self.C['test_vec'].X.shape[0], self.vec.vocab_size))
        # for i in range(len(self.C['test_vec'].X)) :
        #     X_tf[i, self.C['test_vec'].X[i, :]] = 1.

        # X_tf = X_tf[:, 2:]
        # self.C['test_vec'].X = X_tf
        
        df = pd.read_csv('../data/files/decision_aids_filter.csv')
        nb_studies = len(df)
        H = np.zeros((nb_studies, nb_studies))
        cdnos = list(set(df.IM_population))
        for i in range(nb_studies) :
            H[i, df[df['IM_population'] == df['IM_population'][i]].index] = 1
        np.fill_diagonal(H, 0)
        
        self.C['da_ref'] = H
        self.C['da_cdnos'] = df.IM_population
        self.C['da_idxs'] = np.array(df.index)
        self.C['da_vec'] = pickle.load(open('../data/vectorizers/decision_aids_vec_5000.p', 'rb'))

        # X_tf = np.zeros((self.C['da_vec'].X.shape[0], self.vec.vocab_size))
        # for i in range(len(self.C['da_vec'].X)) :
        #     X_tf[i, self.C['da_vec'].X[i, :]] = 1.

        # X_tf = X_tf[:, 2:]
        # self.C['da_vec'].X = X_tf
        

    def common_build_model(self) :
        aspect = self.C['aspect']
        aspect_comp = list(self.C['inputs'])
        aspect_comp.remove('abstract')
        aspect_comp.remove(aspect)
        self.C['aspect_comp'] = aspect_comp
        
        self.modifier = ['S', 'V', 'C']
        self.fields = []
        for input in self.C['inputs'] :
            for mod in self.modifier :
                self.fields += [(mod + input[0].upper(), input)]        
        
        self.fields_in_train = zip(*self.fields)[0]

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
        ce = ModelCheckpoint(weight_str.format('val_loss'),
                             monitor='val_loss',
                             mode='min')
        
        train_idxs, val_idxs = self.C['train_idxs'], self.C['val_idxs']
        
        X = {input: self.C[input].X for input in self.C['inputs']}
        X_train = {input: X_[train_idxs] for input, X_ in X.items()}
        X_val = {input: X_[val_idxs] for input, X_ in X.items()}

        train_cdnos = self.C['cdnos'].iloc[train_idxs].reset_index(drop=True)
        val_cdnos = self.C['cdnos'].iloc[val_idxs].reset_index(drop=True)
        
        batch = bg2(X_val, val_cdnos, nb_sample=self.C['batch_size'])
        
        X_test = {'abstract': self.C['test_vec'].X[self.C['test_idxs']]}
        test_cdnos = self.C['test_cdnos'].reset_index(drop=True)
        batch_test = bg2(X_test, test_cdnos, nb_sample=self.C['batch_size'])
        
        batch_size, nb_epoch = self.C['batch_size'], self.C['nb_epoch']

        
        ss = StudySimilarityLogger(next(batch), self, batch_size=batch_size, logname='val_similarity')
        sst = StudySimilarityLogger(next(batch_test), self, batch_size=batch_size, logname='test_similarity')

        es = EarlyStopping(monitor='val_loss', patience=2, verbose=2, mode='min')
        fl = Flusher()
        cv = CSVLogger(exp_group, exp_id)
        
        al = AUCLogger(self.C['test_vec'].X, self.C['test_ref'], self, batch_size=batch_size)
        pal = PerAspectAUCLogger(self.C['da_vec'].X, self.C['da_ref'], self, batch_size=batch_size)
        
        tb = TensorBoard()
        
        #zlw = LossWeightCallback(self)

        callback_dict = {'cb': cb, # checkpoint best
                         'ce': ce, # checkpoint every
                         'fl': fl, # flusher
                         'es': es, # early stopping
                         'ss': ss, # study similarity logger
                         'st': sst,
                         'cv': cv, # should go *last* as other callbacks populate `logs` dict
                         'al': al
        }
        callback_list = self.C['callbacks'].split(',')
        self.callbacks = [pal]+[callback_dict[cb_name] for cb_name in callback_list]+[tb]#+[zlw]
        
        gen_source_target_batches = bg1(X_train, train_cdnos, self, nb_sample=batch_size)
        gen_validation_batches = bg1(X_val, val_cdnos, self, nb_sample=batch_size)

        nb_train = len(train_idxs)
        self.model.fit_generator(gen_source_target_batches,
                                 steps_per_epoch=(nb_train/batch_size),
                                 epochs=nb_epoch,
                                 verbose=2,
                                 callbacks=self.callbacks,
                                 validation_steps=2,
                                 validation_data = gen_validation_batches)
