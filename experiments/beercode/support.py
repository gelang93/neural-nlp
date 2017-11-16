import sys

import numpy as np
import pandas as pd

import keras.backend as K
from keras.regularizers import l2, l1
from keras.models import Model
from keras.engine.topology import Layer

from keras.layers import Conv1D, MaxPooling1D, Flatten, Input, Permute, Activation, Dense, Lambda, TimeDistributed
from keras.layers.merge import concatenate, Multiply
from keras.layers.core import Dropout

from keras.layers import LeakyReLU

def norm(x, axis=1, keepdims=True):
    return K.sqrt(K.sum(K.square(x), axis=axis, keepdims=keepdims))

def cnn_embed(embedding_layer, filter_lens, nb_filter, max_doclen, word_dim, reg, name):   
    activations = [0]*len(filter_lens)
    convolutions = []
    for i, filter_len in enumerate(filter_lens):
        convolved = Conv1D(nb_filter, 
                           filter_len, 
                           activation='relu',
                           kernel_regularizer=l2(reg))(embedding_layer)
        #convolved = LeakyReLU(alpha=0.01)(convolved)
        convolutions.append(convolved)
        max_pooled = MaxPooling1D(pool_size=max_doclen-filter_len+1)(convolved) # max-1 pooling
        flattened = Flatten()(max_pooled)

        activations[i] = flattened

    concat = concatenate(activations, name=name) if len(filter_lens) > 1 else flattened
    convolutions = concatenate(convolutions, axis=1) if len(filter_lens) > 1 else convolved
    concat = Dropout(0.0)(concat)
    return concat, convolutions

def gated_cnn(lookup, kernel_size, nb_filter, reg) :
    convolved = Conv1D(nb_filter, kernel_size, activation='linear', padding='same', kernel_regularizer=l2(reg))(lookup)
    gates = Conv1D(nb_filter, kernel_size, activation='relu', padding='same', kernel_regularizer=l2(reg))(lookup)
    return Multiply()([convolved, gates])

def gated_cnn_joint(lookup, kernel_size, nb_filter, nb_aspect, reg) :
    nets = []
    gates = []
    convolved = Conv1D(nb_filter, kernel_size, activation='linear', padding='same', kernel_regularizer=l2(reg))(lookup)
    for i in range(nb_aspect) :
        gate = Conv1D(nb_filter, kernel_size, activation='relu', padding='same', kernel_regularizer=l2(reg))(lookup)
        nets.append(Multiply()([convolved, gate]))
        gates.append(gate)
    return nets, gates


def gated_cnn_joint_softmax(lookup, kernel_size, nb_filter, nb_aspect, reg) :
    nets = []
    convolved = Conv1D(nb_filter, kernel_size, activation='linear', kernel_regularizer=l2(reg))(lookup)

    for i in range(nb_aspect) :
        gates = Conv1D(nb_filter, kernel_size, activation='sigmoid', kernel_regularizer=l2(reg))(lookup)
        nets.append(Multiply()([gates, convolved]))

    softmax_gates = Conv1D(nb_aspect, kernel_size, activation='linear', kernel_regularizer=l2(reg))(lookup)
    softmax_gates = Activation('softmax')(softmax_gates)

    for i in range(nb_aspect) :
        net = nets[i]
        gate = Lambda(lambda s : K.expand_dims(s[:,:,i]))(softmax_gates)
        nets[i] = Multiply()([net, gate])

    return nets, nets

def gated_cnn_joint_sub(lookup, kernel_size, nb_filter, nb_aspect, reg) :
    nets = []
    convolved = Conv1D(nb_filter, kernel_size, activation='linear', kernel_regularizer=l2(reg))(lookup)
    gate = Conv1D(nb_filter, kernel_size, activation='sigmoid', kernel_regularizer=l2(reg))(lookup)
    gates = [gate, Lambda(lambda s : 1 - s)(gate)]
    for i in range(nb_aspect) :
        nets.append(Multiply()([convolved, gates[i]]))
    return nets