import os
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score

from keras.callbacks import Callback
import keras.backend as K

import copy

def makedirs(path) :
    d = os.path.dirname(path)
    if not os.path.exists(d) :
        os.makedirs(d)


class Flusher(Callback):
    def on_epoch_end(self, epoch, logs={}):
        sys.stdout.flush()

class CSVLogger(Callback):
    def __init__(self, exp_group, exp_id):
        self.exp_group, self.exp_id = exp_group, exp_id
        self.train_path = '../store/train/{}/{}/{}.csv'.format(self.exp_group, self.exp_id, 'output')    
        makedirs(self.train_path)

        super(Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        frame = {metric: [val] for metric, val in logs.items()}
        print {k:v for k,v in logs.items() if 'loss' not in k}
        pd.DataFrame(frame).to_csv(self.train_path,
                                   index=False,
                                   mode='a' if epoch > 0 else 'w', # overwrite if starting anew if starting anwe
                                   header=epoch==0)

        
class AUCLogger(Callback):
    def __init__(self, X, R, trainer, batch_size=128, phase=0, logname='test_auc'):
        super(Callback, self).__init__()
        self.X = X
        self.R = R
        self.phase = phase
        self.trainer = trainer
        self.nb_sample = len(self.X)
        self.logname = logname
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.embed_studies = self.trainer.construct_evaluation_model(self.model)

    def on_epoch_end(self, epoch, logs={}):
        vecs = []
        i, bs = 0, self.batch_size
        while i*bs < self.nb_sample:
            result = self.embed_studies([self.X[i*bs:(i+1)*bs], self.phase])[0]
            vecs.append(result)
            i += 1
        result = np.concatenate(vecs, axis=0)
        result = normalize(result, 'l2')
        scores = np.dot(result, result.T)
        scores[np.arange(self.nb_sample), np.arange(self.nb_sample)] = -1000
        aucs = [0] * self.nb_sample
        for i in range(self.nb_sample) :
            aucs[i] = roc_auc_score(self.R[i], scores[i])
        logs[self.logname] = np.mean(aucs)
        print logs[self.logname]

class PerAspectAUCLogger(Callback) :
    def __init__(self, X, idxs, trainer, batch_size=128, phase=0, logname='sim') :
        super(Callback, self).__init__()
        self.X = X[idxs]
        self.idxs = idxs
        self.trainer = trainer
        self.nb_sample = len(idxs)
        self.idxs = idxs
        self.logname = logname
        self.batch_size = batch_size

    def create_bit_matrix(self) :
        ds = self.trainer.ds
        aspect_columns = self.trainer.aspect_columns
        idxs = self.idxs
        self.H = {}
        for i in range(4) :
            self.H[str(i)] = np.zeros((len(idxs), len(idxs)))
            a0 = set(ds[ds[aspect_columns[i]] == 0].index) & set(idxs)
            a1 = set(ds[ds[aspect_columns[i]] == 1].index) & set(idxs)
            a0 = map(lambda s : list(idxs).index(s), a0)
            a1 = map(lambda s : list(idxs).index(s), a1)
            for j in a0 :
                self.H[str(i)][j, a0] = 1
            for j in a1 :
                self.H[str(i)][j, a1] = 1
        
            self.H[str(i)][np.arange(len(idxs)), np.arange(len(idxs))] = 0

    def on_train_begin(self, logs={}) :
        self.create_bit_matrix()
        self.embed_studies, self.aspect_embeds = self.trainer.construct_evaluation_model(self.model, aspect_specific=True)

    def on_epoch_end(self, epoch, logs={}) :
        vecs = {k:[] for k in self.aspect_embeds}
        i, bs = 0, self.batch_size
        while i*bs < self.nb_sample:
            for aspect in self.aspect_embeds :
                result = self.aspect_embeds[aspect]([self.X[i*bs:(i+1)*bs], 0])[0]
                vecs[aspect].append(result)
            i += 1
        print ""
        
        for aspect in self.aspect_embeds :
            result = np.concatenate(vecs[aspect], axis=0)
            vecs[aspect] = normalize(result, 'l2')
        
        for aspect in self.aspect_embeds :
            result = vecs[aspect]
            scores = np.dot(result, result.T)
            scores[np.arange(self.nb_sample), np.arange(self.nb_sample)] = -1000
            for aspect_j in self.aspect_embeds :
                aucs = [0] * self.nb_sample
                for i in range(self.nb_sample) :
                    aucs[i] = roc_auc_score(self.H[aspect_j][i], scores[i])
                logs[self.logname+'_'+aspect+'_'+aspect_j] = np.mean(aucs)
                print aspect, aspect_j, logs[self.logname+'_'+aspect+'_'+aspect_j]
        print ""