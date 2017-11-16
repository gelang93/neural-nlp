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
    """Callback that flushes stdout after every epoch"""

    def on_epoch_end(self, epoch, logs={}):
        sys.stdout.flush()

class StudySimilarityLogger(Callback):
    """Callback for computing and inserting the study similarity during training
    
    The study similarity is defined as the mean similarity between studies in
    the same review minus the mean similarity between studies in different
    reviews.
    
    """
    def __init__(self, X, trainer, batch_size=128, phase=0, logname='val_similarity'):
        super(Callback, self).__init__()

        self.X_source = np.concatenate([X['SA'], X['SA']])
        self.X_target = np.concatenate([X['VA'], X['CA']])
        self.phase = phase
        self.nb_sample = len(self.X_source)
        self.batch_size = batch_size
        self.logname = logname
        self.trainer = trainer
        assert len(self.X_target) == self.nb_sample

    def on_train_begin(self, logs={}):
        self.embed_studies = self.trainer.construct_evaluation_model(self.model)

    def on_epoch_end(self, epoch, logs={}):
        source_vecs = []
        target_vecs = []
        i, bs = 0, self.batch_size
        while i*bs < self.nb_sample:
            result = self.embed_studies([self.X_source[i*bs:(i+1)*bs], self.phase])[0]
            source_vecs.append(result)
            target_vecs.append(self.embed_studies([self.X_target[i*bs:(i+1)*bs], self.phase])[0])
            i += 1
            
        source_vecs = np.concatenate(source_vecs, axis=0)
        target_vecs = np.concatenate(target_vecs, axis=0)

        # Get rid of NaNs, and normalize source and target vectors
        source_vecs = normalize(source_vecs, 'l2')
        target_vecs = normalize(target_vecs, 'l2')

        # Compute similarity score
        score = np.sum(source_vecs*target_vecs, axis=1)
        same_study_mean = score[:self.nb_sample/2].mean()
        different_study_mean = score[self.nb_sample/2:].mean()
        logs[self.logname] = same_study_mean / different_study_mean

class CSVLogger(Callback):
    """Callback for dumping csv data during training"""

    def __init__(self, exp_group, exp_id):
        self.exp_group, self.exp_id = exp_group, exp_id
        self.train_path = '../store/train/{}/{}/{}.csv'.format(self.exp_group, self.exp_id, 'output')    
        makedirs(self.train_path)

        super(Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        """Add a line to the csv logging
        
        This csv contains only numbers related to training and nothing regarding
        hyperparameters.
        
        """
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
        #result = self.embed_studies([self.X, self.phase])[0]
        result = np.concatenate(vecs, axis=0)
        result = normalize(result, 'l2')
        scores = np.dot(result, result.T)
        scores[np.arange(self.nb_sample), np.arange(self.nb_sample)] = -1000
        aucs = [0] * self.nb_sample
        for i in range(self.nb_sample) :
            aucs[i] = roc_auc_score(self.R[i], scores[i])
        logs[self.logname] = np.mean(aucs)
        
class PerAspectAUCLogger(Callback) :
    def __init__(self, X, R, trainer, batch_size=128, phase=0, logname='per_aspect_auc') :
        super(Callback, self).__init__()
        self.X,self.R = X,R
        self.phase = phase
        self.trainer = trainer
        self.nb_sample = len(self.X)
        self.logname = logname
        self.batch_size = batch_size
        
    def on_train_begin(self, logs={}) :
        self.ev, self.aspect_embeds = self.trainer.construct_evaluation_model(self.model, aspect_specific=True)
        
    def on_epoch_end(self, epoch, logs={}) :
        for aspect in self.aspect_embeds :
            vecs = []
            i, bs = 0, self.batch_size
            while i*bs < self.nb_sample:
                result = self.aspect_embeds[aspect]([self.X[i*bs:(i+1)*bs], self.phase])[0]
                vecs.append(result)
                i += 1
            #result = self.embed_studies([self.X, self.phase])[0]
            result = np.concatenate(vecs, axis=0)
            result = normalize(result, 'l2')
            scores = np.dot(result, result.T)
            scores[np.arange(self.nb_sample), np.arange(self.nb_sample)] = -1000
            aucs = [0] * self.nb_sample
            for i in range(self.nb_sample) :
                aucs[i] = roc_auc_score(self.R[i], scores[i])
            logs[self.logname + '_' + aspect] = np.mean(aucs)
        
        
        
class LossWeightCallback(Callback) :
    def __init__(self, trainer) :
        super(Callback, self).__init__()
        self.losses = trainer.losses
        self.loss_weights = trainer.loss_weights
        self.zero_after = trainer.zero_after
        self.zero_what = trainer.zero_what
        
    def on_train_begin(self, logs={}) :
        for loss in self.loss_weights :
            self.loss_weights[loss] = 1.0
        self.model.compile(optimizer='adam', loss=self.losses, loss_weights=self.loss_weights)
    
    def on_epoch_end(self, epoch, logs={}) :
        if epoch != self.zero_after :
            return
        print "Zeroing ...." , self.zero_what
        for loss in self.zero_what :
            self.loss_weights[loss] = 0.0
        self.model.compile(optimizer='adam', loss=self.losses, loss_weights=self.loss_weights)