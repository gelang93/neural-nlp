import sys

from functools import partial

import numpy as np
import pandas as pd

import keras.backend as K
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2
from keras.layers import Flatten
from keras.models import Model

from metrics import compute_f1, compute_acc


def norm(x):
    """Compute the frobenius norm of x

    Parameters
    ----------
    x : a keras tensor

    """
    return K.sqrt(K.sum(K.square(x)))

def per_class_f1s(ys):
    """Build up a list of callbacks to compute the f1 for each class and macro f1

    Parameters
    ----------
    ys : labels for training data

    """
    classes = pd.Series(ys.argmax(axis=1)).unique()
    classes, num_classes = np.sort(classes), len(classes)

    f1s = [0]*num_classes
    for i in classes:
        keras_f1_wrapper = partial(compute_f1, i)
        keras_f1_wrapper.__name__ = 'f1_{}'.format(i) # keras compile demands __name__ be set
        
        f1s[i] = keras_f1_wrapper

    def macro_f1(y_true, y_pred):
        numer = f1s[0](y_true, y_pred) + f1s[1](y_true, y_pred) + f1s[2](y_true, y_pred)
        return numer / 3

    return f1s + [macro_f1]

def per_class_accs(ys, multi_label=True):
    """Build up a list of callbacks to compute the per-class accuracies

    Parameters
    ----------
    ys : labels for training data
    multi_label : True if ys has multiple labels per training example

    """
    nb_train, nb_class = ys.shape
    classes = np.arange(nb_class, dtype=np.int)

    accs = [0]*nb_class
    for i in classes:
        keras_acc_wapper = partial(compute_acc, i, multi_label)
        keras_acc_wapper.__name__ = 'acc_{}'.format(i) # keras compile demands __name__ be set
        
        accs[i] = keras_acc_wapper

    def macro_acc(y_true, y_pred):
        sum = K.variable(value=0.) # index of class to compute
        for i in range(nb_class):
            sum += accs[i](y_true, y_pred)
        return sum / nb_class

    return accs + [macro_acc] + ['accuracy'] # have keras compute micro-accuracy also

def average(inputs):
    """Merge function which averages all of the input tensors
    
    All tensors must have the same shape. The output shape is unchanged.
    
    """
    accum = K.zeros_like(inputs[0])
    for input in inputs:
        accum += input
    return accum / float(len(inputs))

def get_trainable_weights(model):
    """Find all layers which are trainable in the model

    Surprisingly `model.trainable_weights` will return layers for which
    `trainable=False` has been set, hence the extra check.

    """
    tensors = [tensor for tensor in model.trainable_weights if model.get_layer(tensor.name[:-2]).trainable]
    names = [tensor.name for tensor in tensors]

    return names, tensors

def stratified_batch_generator(X_train, y_train, batch_size, mb_ratios, num_classes):
    """Make a genertor suitable for keras.Model.fit_generator() to consume

    Define it in the scope of arguments to this function

    """
    y_train = y_train.argmax(axis=1)

    classes = pd.Series(y_train).unique()
    classes = np.sort(classes)
    assert classes[0] == 0 and classes[1] == 1 and classes[2] == 2

    class_idxs = {}
    for class_ in classes:
        class_idxs[class_] = np.argwhere(y_train == class_).flatten()

    while True:
        # do stratified sampling
        mb_idxs = [0]*num_classes
        for i, (class_, mb_ratio) in enumerate(zip(classes, mb_ratios)):
            nb_idxs = np.ceil(batch_size*mb_ratio).astype(np.int)
            mb_idxs[i] = np.random.choice(class_idxs[class_], size=nb_idxs)

        batch_idxs = np.concatenate(mb_idxs)[:batch_size]

        yield X_train[batch_idxs], to_categorical(y_train[batch_idxs])

def cnn_embed(embedding_layer, filter_lens, nb_filter, max_doclen, word_dim, reg, name):
    """Add conv -> max_pool -> flatten for each filter length
    
    Parameters
    ----------
    words : tensor of shape (max_doclen, vector_dim)
    filter_lens : list of n-gram filers to run over `words`
    nb_filter : number of each ngram filters to use
    max_doclen : length of the document
    reg : regularization strength
    name : name to give the merged vector
    
    """
    from keras.layers import Conv1D, MaxPooling1D, Flatten, Input
    from keras.layers.merge import concatenate

    activations = [0]*len(filter_lens)
    for i, filter_len in enumerate(filter_lens):
        convolved = Conv1D(nb_filter, 
                           filter_len, 
                           activation='relu',
                           kernel_regularizer=l2(reg))(embedding_layer)

        max_pooled = MaxPooling1D(pool_size=max_doclen-filter_len+1)(convolved) # max-1 pooling
        flattened = Flatten()(max_pooled)

        activations[i] = flattened

    concat = concatenate(activations, name=name) if len(filter_lens) > 1 else flattened
    #model = Model(words, concat)
    return concat

def top_reviews(cdnos, k):
    top_cdnos = set(cdnos.value_counts().sort_values(ascending=False)[:k])
    top_idxs = cdnos.map(lambda cdno: cdno in top_cdnos).index

    return np.array(top_idxs)

    # get study indices to train on based on k most popular reviews
    df = df[df.cdno.map(lambda cdno: cdno in top_cdnos)] # throw out studies we are not using
    cdnos = df.reset_index(drop=True).cdno # get cdnos so we can map studies to their cdno
    train_idxs, val_idxs = np.array((df.train == True).index), np.array((df.train == True).index)

def norm2d(x):
    """Take the norm of a 2D tensor

    Parameters
    ----------
    x : a batch of 3D tensors

    To be used in a Lambda layer.

    """
    flat = Flatten()(x)
    squared = K.square(flat)
    summed = K.sum(squared, axis=1, keepdims=True)
    return summed
