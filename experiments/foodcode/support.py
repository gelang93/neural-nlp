import sys

import numpy as np
import pandas as pd

import keras.backend as K
from keras.regularizers import l2, l1
from keras.models import Model
from keras.engine.topology import Layer

from keras.layers import Conv1D, MaxPooling1D, Flatten, Input, Permute, Activation, Dense, Lambda
from keras.layers.merge import concatenate
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
    #concat = Dense(nb_filter*3, activation='tanh', kernel_regularizer=l2(reg))(concat)
    return concat, convolutions

class Bilinear(Layer) :
    def __init__(self, **kwargs) :
        super(Bilinear, self).__init__(**kwargs)
    
    def build(self, input_shape) :
        print input_shape
        self.W = self.add_weight('W', 
                                 shape=(input_shape[0][1], input_shape[1][1]), 
                                 initializer='uniform',
                                 trainable=True)
        super(Bilinear, self).build(input_shape)
        
    def call(self, x) :
        return K.batch_dot(x[1], K.dot(x[0], self.W),  axes=-1)
    
    def compute_output_shape(self, input_shape) :
        return (input_shape[0][0], 1)
    
class Element_wise_weighting(Layer) :
    def __init__(self, nb_aspects, **kwargs) :
        self.nb_aspects = nb_aspects
        super(Element_wise_weighting, self).__init__(**kwargs)
    
    def build(self, input_shape) :
        self.W = self.add_weight('W', 
                                 shape=(input_shape[1], self.nb_aspects), 
                                 initializer='uniform',
                                 trainable=True)
        super(Element_wise_weighting, self).build(input_shape)
        
    def call(self, x) :
        x1 = K.repeat(x, self.nb_aspects)
        x2 = K.permute_dimensions(x1, (0, 2, 1))
        return x2 * K.softmax(x2 * self.W)
    
    def compute_output_shape(self, input_shape) :
        return tuple(list(input_shape) + [self.nb_aspects])
