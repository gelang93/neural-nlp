import sys

from functools import partial

import numpy as np
import pandas as pd

import keras.backend as K
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2
from keras.layers import Flatten
from keras.models import Model

def norm(x):
    """Compute the frobenius norm of x

    Parameters
    ----------
    x : a keras tensor

    """
    return K.sqrt(K.sum(K.square(x)))

def get_trainable_weights(model):
    """Find all layers which are trainable in the model

    Surprisingly `model.trainable_weights` will return layers for which
    `trainable=False` has been set, hence the extra check.

    """
    tensors = [tensor for tensor in model.trainable_weights if model.get_layer(tensor.name[:-2]).trainable]
    names = [tensor.name for tensor in tensors]

    return names, tensors

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
    from keras.layers.core import Dropout

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
    concat = Dropout(0.5)(concat)
    return concat

def lstm_embed(embedding_layer, hidden_dim, reg) :
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import Bidirectional
    from keras.layers.core import Dropout
    lstm = Bidirectional(LSTM(hidden_dim, dropout=0.5, recurrent_dropout=0.2))(embedding_layer)
    return lstm
