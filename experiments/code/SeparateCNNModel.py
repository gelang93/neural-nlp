from collections import OrderedDict

import keras.backend as K
from keras.layers import Input, Embedding, Concatenate
from keras.layers.merge import Dot
from keras.models import Model
from keras.regularizers import l2

from trainer import Trainer
from support import cnn_embed, lstm_embed

import numpy as np

class SeparateCNNModel(Trainer):
    def build_model(self, nb_filter=300, filter_lens=range(1,6), reg=0.0001):
        aspect = self.C['aspect'][0].upper()
        inputs = ['SA']
        scores = []
        # for mod in ['valid', 'corrupt'] :
        #     inputs += [mod + '_abstract']
        #     scores += [mod + '_abstract']
            
        for mod in self.modifier :
            inputs += [mod + aspect]
            scores += [mod + aspect]
        
        for mod in ['S'] :
            for aspect in self.C['aspect_comp'] :
                inputs += [mod + aspect[0].upper()]
                scores += [mod + aspect[0].upper()]
        
        maxlen = self.vec.maxlen
        word_dim, vocab_size = self.vec.word_dim, self.vec.vocab_size

        input = Input(shape=[maxlen], dtype='int32')
        lookup = Embedding(output_dim=word_dim, 
                           input_dim=vocab_size, 
                           name='embedding', 
                           weights=[self.vec.embeddings])(input)
        cnn_network = cnn_embed(lookup, filter_lens, nb_filter, maxlen, word_dim, reg, name='pool')
        model = Model(input, cnn_network)
        model.name = 'pool'
        
        I = OrderedDict() # inputs
        for s in inputs:
            I[s] = Input(shape=[maxlen], dtype='int32', name=s)
        
        C = OrderedDict()
        for s in inputs:
            C[s] = model(I[s])

        D = OrderedDict() # dots
        for s in scores:
            D[s] = Dot(axes=1, name=s+'_score')([C['same_abstract'], C[s]])

        self.model = Model(inputs=I.values(), outputs=D.values())
        losses = {}
        for s in scores :
            losses[s + '_score'] = 'hinge'
        
        self.model.compile(optimizer='adam', loss=losses)
        
    def generate_y_batch(self, nb_sample) :
        aspect = self.C['aspect'][0].upper()
        y_batch = {}
        
        y_batch['S' + aspect + '_score'] = np.ones(nb_sample)
        y_batch['V' + aspect + '_score'] = np.ones(nb_sample)
        y_batch['C' + aspect + '_score'] = np.full(shape=nb_sample, fill_value=-1)
        
        y_batch['CA_score'] = np.full(shape=nb_sample, fill_value=-1)
        y_batch['VA_score'] = np.ones(nb_sample)
        
        y_batch['V' + aspect + '_aspect_score'] = np.ones(nb_sample)
        y_batch['C' + aspect + '_aspect_score'] = np.full(shape=nb_sample, fill_value=-1)

        for a in self.C['aspect_comp'] :
            for mod in self.modifier :
                y_batch[mod + a[0].upper() + '_score'] = np.full(shape=nb_sample, fill_value=-1)
            
        return y_batch
    
    def construct_evaluation_model(self, model) :
        inputs = model.get_layer('pool').inputs
        inputs += [K.learning_phase()]
        output = model.get_layer('pool').get_output_at(0)
        self.evaluation_model = K.function(inputs, [output])
        return self.evaluation_model