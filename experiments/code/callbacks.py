import os
import sys

import pickle

import numpy as np
import pandas as pd

import scipy
from sklearn.metrics import f1_score

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

class ValidationLogger(Callback):
    """Use to test that `metrics.compute_f1` is implemented correctly"""

    def __init__(self, X_val, y_val):
        super(Callback, self).__init__()
        self.X_val, self.y_val = X_val, y_val.argmax(axis=1)

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val).argmax(axis=1)

        f1s = f1_score(self.y_val, y_pred, average=None)

        print 'scikit f1s:', f1s
        print 'scikit f1:', np.mean(f1s)

class PrecisionLogger(Callback):
    """Callback for computing precision during training"""

    def __init__(self, X, study_dim, batch_size=128, phase=0):
        """Save variables and sample study indices

        Parameters
        ----------
        X_source : vectorized studies
        X_target : vectorized studies where X_target[i] has the same cdno as X_source[i]
        cdnos : mapping from study indexes to their cdno
        nb_sample : number of studies to evaluate
        phase : 1 for train and 0 for test

        """
        super(Callback, self).__init__()

        self.X_source = np.concatenate([X['same_abstract'], X['same_abstract']])
        self.X_target = np.concatenate([X['valid_abstract'], X['corrupt_abstract']])
        self.phase = phase
        self.nb_sample = len(self.X_source)
        self.study_dim = study_dim
        self.batch_size = batch_size
        assert len(self.X_target) == self.nb_sample

    def on_train_begin(self, logs={}):
        """Build keras function to produce vectorized studies
        
        Even though some tensors may not need all of these inputs, it doesn't
        hurt to include them for those that do.
        
        """
        # build keras function to get study embeddings
        inputs = [self.model.inputs[0], K.learning_phase()]
        outputs = self.model.get_layer('pool').output
        self.embed_studies = K.function(inputs, [outputs])

    def on_epoch_begin(self, epoch, logs={}):
        """Compute precision between studies in same and different reviews"""

        source_vecs = np.zeros([len(self.X_source), self.study_dim])
        target_vecs = np.zeros([len(self.X_target), self.study_dim])
        i, bs = 0, self.batch_size
        while i*bs < self.nb_sample:
            result = self.embed_studies([self.X_source[i*bs:(i+1)*bs], self.phase])[0]
            source_vecs[i*bs:(i+1)*bs] = result
            target_vecs[i*bs:(i+1)*bs] = self.embed_studies([self.X_target[i*bs:(i+1)*bs], self.phase])[0]
            i += 1

        # Get rid of any NaNs (shouldn't have to have this but yolo)
        source_vecs[np.isnan(source_vecs)] = 0
        target_vecs[np.isnan(target_vecs)] = 0

        # Compute similarity score
        score = np.sum(source_vecs*target_vecs, axis=1)
        logs['val_precision'] = np.mean(score[:self.nb_sample/2] > score[self.nb_sample/2:])

class StudySimilarityLogger(Callback):
    """Callback for computing and inserting the study similarity during training
    
    The study similarity is defined as the mean similarity between studies in
    the same review minus the mean similarity between studies in different
    reviews.
    
    """
    def __init__(self, X, study_dim, batch_size=128, phase=0):
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
        self.phase = phase
        self.nb_sample = len(self.X_source)
        self.study_dim = study_dim
        self.batch_size = batch_size
        assert len(self.X_target) == self.nb_sample

    def on_train_begin(self, logs={}):
        """Build keras function to produce vectorized studies
        
        Even though some tensors may not need all of these inputs, it doesn't
        hurt to include them for those that do.
        
        """
        # build keras function to get study embeddings
        inputs = [self.model.inputs[0], K.learning_phase()]
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
        source_vecs[np.isnan(source_vecs)] = 0
        target_vecs[np.isnan(target_vecs)] = 0
        source_norms = np.apply_along_axis(np.linalg.norm, axis=1, arr=source_vecs)
        target_norms = np.apply_along_axis(np.linalg.norm, axis=1, arr=target_vecs)
        source_vecs = source_vecs / source_norms[:, np.newaxis]
        target_vecs = target_vecs / target_norms[:, np.newaxis]

        # Compute similarity score
        score = np.sum(source_vecs*target_vecs, axis=1)
        same_study_mean = score[:self.nb_sample/2].mean()
        different_study_mean = score[self.nb_sample/2:].mean()
        logs['val_similarity'] = same_study_mean / different_study_mean
        print logs['val_similarity']

class TensorLogger(Callback):
    """Callback for monitoring value of tensors during training"""

    def __init__(self, X, y, exp_group, exp_id, tensor_funcs, phase=1):
        """Save variables

        Parameters
        ----------
        X : training data
        y : training labels
        tensor_funcs : list of functions which take a keras model and produce tensors as well as their names
        exp_group : experiment group
        exp_id : experiment id
        phase : 0 for test phase and 1 for learning phase (defaults to 1 because
        we may want to examine number of neurons surviving dropout, for instance)

        Note: `tensor_funcs` is a hack. Ideally we would just pass the tensors
        themselves, but tensors owned by the optimizer (e.g. update tensors)
        don't become available until keras does some magic in fit(). Hence we
        work around this by providing functions that will return tensors instead
        of the tensors themselves.

        """
        super(Callback, self).__init__()

        self.phase = phase
        self.X, self.y = X, y
        self.tensor_funcs = tensor_funcs
        self.exp_group = exp_group
        self.exp_id = exp_id

        self.nb_sample = len(X[0])
        assert len(X[1]) == self.nb_sample

        self.tensors, self.names = [], []

    def on_train_begin(self, logs={}):
        """Build keras function to evaluate all tensors in one call
        
        Even though some tensors may not need all of these inputs, it doesn't
        hurt to include them for those that do.
        
        """
        self.names, self.tensors = [], []
        for tensor_func in self.tensor_funcs:
            names, tensors = tensor_func(self.model)
            self.names, self.tensors = self.names+names, self.tensors+tensors

        # append suffix to enableidentification in `logs` by other callbacks
        self.names = [name+'_tensor' for name in self.names]

        inputs = self.model.inputs
        labels, sample_weights = self.model.targets[0], self.model.sample_weights[0] # needed to compute gradient/update tensors
        learning_phase = K.learning_phase() # needed to compute forward pass (e.g. dropout)
        self.eval_tensors = K.function(inputs=inputs+[labels, sample_weights, learning_phase], outputs=self.tensors)

    def on_epoch_end(self, epoch, logs={}):
        """Evaluate tensors and log their values
        
        Take a small subset of the training data to run through the network to
        compute this tensors. The subset differs each epoch.
        
        """
        tensor_vals = self.eval_tensors(self.X + [self.y,
                                                  np.ones(self.nb_sample), # each sample has the same weight
                                                  self.phase])

        for tensor_val, name in zip(tensor_vals, self.names):
            nb_nan = np.isnan(tensor_val).sum()
            if nb_nan == 0:
                continue

            print '{} has {} NaNs!'.format(name, nb_nan)
            self.model.stop_training = True # end training prematurely

        if not self.model.stop_training:
            return

        # dump all tensor vals
        data_fname = '../store/nan-snapshot/{}/{}/{}.p'.format(self.exp_group, self.exp_id, {})
        makedirs(data_fname)
            
        for tensor_val, name in zip(tensor_vals, self.names):
            filename = data_fname.format(name)
            pickle.dump(np.array(tensor_val), open(filename, 'wb'))

        # dump model
        self.model.save('../store/nan-snapshot/{}/{}/{}.h5'.format(self.exp_group, self.exp_id, 'model'))

        # dump batch
        pickle.dump(self.X, open(data_fname.format(self.exp_group, self.exp_id, 'X'), 'wb'))
        pickle.dump(self.y, open(data_fname.format(self.exp_group, self.exp_id, 'y'), 'wb'))

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
        print(self.train_path)
        pd.DataFrame(frame).to_csv(self.train_path,
                                   index=False,
                                   mode='a' if epoch > 0 else 'w', # overwrite if starting anew if starting anwe
                                   header=epoch==0)

class ProbaLogger(Callback):
    """Callback for dumping info for error-analysis

    Currently dump predicted probabilities for each validation example.

    """
    def __init__(self, exp_group, exp_id, X_val, nb_train, nb_class, batch_size, metric):
        self.exp_group, self.exp_id = exp_group, exp_id
        self.nb_train = nb_train
        self.nb_class = nb_class
        self.X_val = X_val
        self.batch_size = batch_size
        self.metric = metric

        self.best_score = 0
        self.proba_loc = '../store/probas/{}/{}.p'.format(self.exp_group, self.exp_id)
        makedirs(self.proba_loc)
        
        # initally we haven't predicted anything
        if not os.path.exists(self.proba_loc):
            pickle.dump(np.zeros([self.nb_train, self.nb_class]), open(self.proba_loc, 'wb'))

        super(Callback, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        """Update existing predicted probas or add probs from new fold"""

        score = logs[self.metric]
        if score <= self.best_score:
            return

        self.best_score = score
        y_proba = pickle.load(open(self.proba_loc))
        y_proba[self.val_idxs] = self.model.predict(self.X_val, batch_size=self.batch_size)
        pickle.dump(y_proba, open(self.proba_loc, 'wb'))

class StudyLogger(Callback):
    """Callback for dumping study embeddings
    
    Dump study vectors every time we reach a new best for study similarity.
    
    """
    def __init__(self, X_study, exp_group, exp_id):
        """Save variables

        Parameters
        ----------
        X_study : vectorized abstracts
        exp_group : experiment group
        exp_id : experiment id

        """
        super(Callback, self).__init__()

        self.X_study = X_study
        self.exp_group, self.exp_id = exp_group, exp_id

        self.dump_loc = '../store/study_vecs/{}/{}.p'.format(self.exp_group, self.exp_id)
        makedirs(self.dump_loc)
        
        self.max_score = -np.inf # study similarity score (computed in StudySimilarityLogger)

    def on_train_begin(self, logs={}):
        """Define keras function for computing study embeddings"""

        inputs = [self.model.inputs[0], K.learning_phase()] # don't need summary
        outputs = self.model.get_layer('study').output

        self.embed_abstracts = K.function(inputs, [outputs])

    def on_epoch_end(self, epoch, logs={}):
        """Run all abstracts through model and dump the embeddings"""

        score = logs['similarity_score']
        if score < self.max_score:
            return # only log study vectors when we reach a new best similarity score

        TEST_MODE = 0 # learning phase of 0 for test mode (i.e. do *not* apply dropout)
        abstract_vecs = self.embed_abstracts([self.X_study, TEST_MODE])
        pickle.dump(abstract_vecs[0], open(self.dump_loc, 'w'))
