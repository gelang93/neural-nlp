import sys

from functools import partial

import numpy as np
import pandas as pd

import keras.backend as K
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2, l1
from keras.layers import Flatten, Lambda
from keras.models import Model
from keras.layers import LeakyReLU

from keras.engine.topology import Layer

from keras.layers import Conv1D, MaxPooling1D, Flatten, Input, Permute, Activation, PReLU
from keras.layers.merge import concatenate, Multiply
from keras.layers.core import Dropout

def norm(x, axis=1, keepdims=True):
    return K.sqrt(K.sum(K.square(x), axis=axis, keepdims=keepdims))

def get_trainable_weights(model):
    tensors = [tensor for tensor in model.trainable_weights if model.get_layer(tensor.name[:-2]).trainable]
    names = [tensor.name for tensor in tensors]

    return names, tensors

from keras.activations import softmax
softmax_pooling = Lambda(lambda s : K.sum(softmax(s, axis=1)*s, axis=1, keepdims=True))

def cnn_embed(embedding_layer, filter_lens, nb_filter, max_doclen, reg):
    from keras.layers import Conv1D, MaxPooling1D, Flatten, Input, Permute, Activation
    from keras.layers.merge import concatenate
    from keras.layers.core import Dropout

    activations = [0]*len(filter_lens)
    convolutions = []
    for i, filter_len in enumerate(filter_lens):
        convolved = Conv1D(nb_filter, 
                           filter_len, 
                           activation='tanh',
                           kernel_regularizer=l2(reg))(embedding_layer)
        convolutions.append(convolved)
        max_pooled = MaxPooling1D(pool_size=max_doclen-filter_len+1)(convolved) # max-1 pooling
        flattened = Flatten()(max_pooled)

        activations[i] = flattened

    concat = concatenate(activations) if len(filter_lens) > 1 else flattened
    convolutions = concatenate(convolutions, axis=1) if len(filter_lens) > 1 else convolved
    #concat = Dropout(0.3)(concat)
    return concat, convolutions


def gated_cnn(lookup, kernel_size, nb_filter, reg) :
    convolved = Conv1D(nb_filter, kernel_size, activation='linear', padding='same', kernel_regularizer=l2(reg))(lookup)
    convolved = PReLU(shared_axes=[1])(convolved)

    gates = Conv1D(1, kernel_size, 
                        activation='sigmoid',  padding='same',
                        kernel_regularizer=l2(reg),
                        activity_regularizer=l1(reg*0.01))(lookup)

    return convolved, gates


