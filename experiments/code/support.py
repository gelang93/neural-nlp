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

def lstm_embed(embedding_layer, hidden_dim) :
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import Bidirectional
    from keras.layers.core import Dropout
    lstm = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.5, recurrent_dropout=0.2))(embedding_layer)
    return lstm

def gated_cnn(lookup, kernel_size, nb_filter, reg) :
    convolved = Conv1D(nb_filter, kernel_size, activation='linear', padding='same', kernel_regularizer=l2(reg))(lookup)
    convolved = PReLU(shared_axes=[1])(convolved)
    # model_gates = Conv1D(nb_filter, kernel_size, activation="sigmoid", padding='same', kernel_regularizer=l2(reg))(lookup)
    # convolved = Multiply()([convolved, model_gates])
    # convolved_s = Permute((2, 1))(convolved)
    # convolved_s = Activation('softmax')(convolved_s)
    # convolved_s = Permute((2, 1))(convolved_s)
    gates = Conv1D(1, kernel_size, 
                        activation='sigmoid',  padding='same',
                        kernel_regularizer=l2(reg),
                        activity_regularizer=l1(reg*0.01))(lookup)
    #return Multiply()([convolved, gates])
    #return Multiply()([convolved, convolved_s])
    return convolved, gates

def gated_cnn_joint(lookup, kernel_size, nb_filter, nb_aspect, reg) :
    nets = []
    gates = []
    convolved = Conv1D(nb_filter, kernel_size, activation='linear', padding='same', kernel_regularizer=l2(reg))(lookup)
    for i in range(nb_aspect) :
        gate = Conv1D(nb_filter, kernel_size, activation='relu', padding='same', kernel_regularizer=l2(reg))(lookup)
        nets.append(Multiply()([convolved, gate]))
        gates.append(gate)
    return nets, gates
    
def cnn_embed_reshape(embedding_layer, filter_lens, nb_filter, nb_aspect, max_doclen, reg):
    from keras.layers import Conv1D, MaxPooling1D, Flatten, Reshape
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
        flattened = Reshape((nb_aspect, nb_filter/nb_aspect))(flattened)
        activations[i] = flattened

    concat = concatenate(activations, axis=-1) if len(filter_lens) > 1 else flattened
    concat = Dropout(0.5)(concat)
    return concat

def cnn_embed_me(el1, el2, filter_lens, nb_filter, mlen1, mlen2, reg):
    a1, a2 = [0]*len(filter_lens), [0]*len(filter_lens)
    for i, filter_len in enumerate(filter_lens):
        convolved = Conv1D(nb_filter, 
                           filter_len, 
                           activation='relu',
                           kernel_regularizer=l2(reg))
        c1, c2 = convolved(el1), convolved(el2)
        mp1 = MaxPooling1D(pool_size=mlen1-filter_len+1)(c1)
        mp2 = MaxPooling1D(pool_size=mlen2-filter_len+1)(c2)
        f1, f2 = Flatten()(mp1), Flatten()(mp2)
        a1[i], a2[i] = f1, f2

    concat1 = concatenate(a1) if len(filter_lens) > 1 else f1
    concat2 = concatenate(a2) if len(filter_lens) > 1 else f2

    concat1, concat2 = Dropout(0.5)(concat1), Dropout(0.5)(concat2)
    return concat1, concat2
