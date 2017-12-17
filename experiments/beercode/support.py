import sys

import numpy as np
import pandas as pd

import keras.backend as K
from keras.regularizers import l2, l1
from keras.models import Model
from keras.engine.topology import Layer

from keras.layers import Conv1D, MaxPooling1D, Flatten, Input, Permute, Activation, Dense, Lambda, TimeDistributed, PReLU
from keras.layers.merge import concatenate, Multiply
from keras.layers.core import Dropout

def cnn_embed(embedding_layer, filter_lens, nb_filter, max_doclen, reg): 
    filter_lens = [3]  
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

    return concat

def gated_cnn(lookup, kernel_size, nb_filter, reg) :
    convolved = Conv1D(nb_filter, kernel_size, activation='linear', padding='same', kernel_regularizer=l2(reg))(lookup)
    convolved = PReLU(shared_axes=[1])(convolved)
    
    #gates = Conv1D(nb_filter, kernel_size, activation='sigmoid', padding='same', kernel_regularizer=l2(reg))(lookup)

    out_gates = Conv1D(1, kernel_size, 
                        activation='sigmoid', 
                        padding='same', 
                        kernel_regularizer=l2(reg),
                        activity_regularizer=l1(reg))(lookup)
    #return Multiply()([convolved, out_gates]), out_gates
    return convolved, out_gates