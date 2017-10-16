import os
import sys

import pickle

import numpy as np
import pandas as pd

import scipy
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score


from keras.callbacks import Callback
from keras.utils.np_utils import to_categorical
import keras.backend as K

import loggers

import copy

def makedirs(path) :
    d = os.path.dirname(path)
    if not os.path.exists(d) :
        os.makedirs(d)


class Flusher(Callback):
    """Callback that flushes stdout after every epoch"""

    def on_epoch_end(self, epoch, logs={}):
        sys.stdout.flush()

class LargeWordCallback(Callback):
    """Callback that logs the largest words after every epoch"""

    def __init__(self, idx2word, top_n=20):
        super(Callback, self).__init__()
        self.top_n = top_n
        self.idx2word = idx2word

    def on_epoch_end(self, epoch, logs={}):
        W, = self.model.get_layer('embedding').get_weights()
        word_norms = np.sum(W**2, axis=1)
        large_idxs = np.argsort(-word_norms)[:self.top_n]
        large_words = [self.idx2word[idx] for idx in large_idxs]
        logs['words'] = ','.join(large_words)

class StudySimilarityLogger(Callback):
    """Callback for computing and inserting the study similarity during training
    
    The study similarity is defined as the mean similarity between studies in
    the same review minus the mean similarity between studies in different
    reviews.
    
    """
    def __init__(self, X, study_dim, batch_size=128, phase=0, logname='val_similarity'):
        """Save variables and sample study indices

        Parameters
        ----------
        X_source : vectorized studies
        X_target : vectorized studies where X_target[i] has the same cdno as X_source[i]
        study_dim : dimension of study vectors
        cdnos : mapping from study indexes to their cdno
        nb_sample : number of studies to evaluate
        phase : 1 for train and 0 for test

        """
        super(Callback, self).__init__()

        self.X_source = np.concatenate([X['same_abstract'], X['same_abstract']])
        self.X_target = np.concatenate([X['valid_abstract'], X['corrupt_abstract']])
        print self.X_source.shape
        self.phase = phase
        self.nb_sample = len(self.X_source)
        self.study_dim = study_dim
        self.batch_size = batch_size
        self.logname = logname
        assert len(self.X_target) == self.nb_sample

    def on_train_begin(self, logs={}):
        """Build keras function to produce vectorized studies
        
        Even though some tensors may not need all of these inputs, it doesn't
        hurt to include them for those that do.
        
        """
        # build keras function to get study embeddings
        inputs = self.model.get_layer('pool').inputs
        inputs += [K.learning_phase()]
        outputs = self.model.get_layer('pool').get_output_at(0)
        self.embed_studies = K.function(inputs, [outputs])

    def on_epoch_end(self, epoch, logs={}):
        """Compute study similarity from the same review and different reviews"""

        source_vecs = np.zeros([len(self.X_source), self.study_dim])
        target_vecs = np.zeros([len(self.X_target), self.study_dim])
        i, bs = 0, self.batch_size
        while i*bs < self.nb_sample:
            result = self.embed_studies([self.X_source[i*bs:(i+1)*bs], self.phase])[0]
            source_vecs[i*bs:(i+1)*bs] = result
            target_vecs[i*bs:(i+1)*bs] = self.embed_studies([self.X_target[i*bs:(i+1)*bs], self.phase])[0]
            i += 1

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

    def __init__(self, exp_group, exp_id, fold):
        args = sys.argv[2:]
        hyperparam_dict  = dict(args.split('=') for args in args)

        self.exp_group, self.exp_id = exp_group, exp_id
        self.fold = fold
        self.train_path = '../store/train/{}/{}/{}.csv'.format(self.exp_group, self.exp_id, self.fold)    
        makedirs(self.train_path)

        # write out hyperparams to disk
        hp_path = '../store/hyperparams/{}/{}.csv'.format(self.exp_group, self.exp_id)
        makedirs(hp_path)
        
        hp = pd.Series(hyperparam_dict)
        hp.index.name, hp.name = 'hyperparam', 'value'
        hp.to_csv(hp_path, header=True)

        super(Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        """Add a line to the csv logging
        
        This csv contains only numbers related to training and nothing regarding
        hyperparameters.
        
        """
        frame = {metric: [val] for metric, val in logs.items()}
        print logs
        pd.DataFrame(frame).to_csv(self.train_path,
                                   index=False,
                                   mode='a' if epoch > 0 else 'w', # overwrite if starting anew if starting anwe
                                   header=epoch==0)

        
class AUCLogger(Callback):
    """Callback for computing and inserting the study similarity during training
    
    The study similarity is defined as the mean similarity between studies in
    the same review minus the mean similarity between studies in different
    reviews.
    
    """
    def __init__(self, X, R, phase=0, logname='test_auc'):
        """Save variables and sample study indices

        Parameters
        ----------
        X_source : vectorized studies
        X_target : vectorized studies where X_target[i] has the same cdno as X_source[i]
        study_dim : dimension of study vectors
        cdnos : mapping from study indexes to their cdno
        nb_sample : number of studies to evaluate
        phase : 1 for train and 0 for test

        """
        super(Callback, self).__init__()
        self.X = X
        self.R = R
        self.phase = phase
        self.nb_sample = len(self.X)
        self.logname = logname

    def on_train_begin(self, logs={}):
        """Build keras function to produce vectorized studies
        
        Even though some tensors may not need all of these inputs, it doesn't
        hurt to include them for those that do.
        
        """
        # build keras function to get study embeddings
        inputs = self.model.get_layer('pool').inputs
        inputs += [K.learning_phase()]
        outputs = self.model.get_layer('pool').get_output_at(0)
        self.embed_studies = K.function(inputs, [outputs])

    def on_epoch_end(self, epoch, logs={}):
        """Compute study similarity from the same review and different reviews"""

        result = self.embed_studies([self.X, self.phase])[0]
        result = normalize(result, 'l2')
        scores = np.dot(result, result.T)
        scores[np.arange(self.nb_sample), np.arange(self.nb_sample)] = -1000
        aucs = [0] * self.nb_sample
        for i in range(self.nb_sample) :
            aucs[i] = roc_auc_score(self.R[i], scores[i])
        logs[self.logname] = np.mean(aucs)
        
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