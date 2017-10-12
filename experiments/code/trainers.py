from collections import OrderedDict

import keras.backend as K
from keras.layers import Input, Embedding, Dropout, Dense, LSTM, merge, Lambda
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, merge
from keras.layers import Activation, Lambda, ActivityRegularization
from keras.layers.merge import Dot
from keras.models import Model
from keras.regularizers import l2

from trainer import Trainer
from support import cnn_embed, lstm_embed

import numpy as np

class SharedCNNSiameseTrainer(Trainer):
    def build_model(self, nb_filter=300, filter_lens=range(1,4), reg=0.0001):
        aspect = self.C['aspect']
        inputs = ['same_abstract']
        scores = []
        for mod in ['valid', 'corrupt'] :
            inputs += [mod + '_abstract']
            scores += [mod + '_abstract']
            
        for mod in ['same'] :
            inputs += [mod + '_' + aspect]
            scores += [mod + '_' + aspect]
        
        for mod in ['same'] :
            for aspect in self.C['aspect_comp'] :
                inputs += [mod + '_' + aspect]
                scores += [mod + '_' + aspect]
        
        maxlen = self.vec.maxlen
        word_dim, vocab_size = self.vec.word_dim, self.vec.vocab_size

        input = Input(shape=[maxlen], dtype='int32')
        lookup = Embedding(output_dim=word_dim, 
                           input_dim=vocab_size, 
                           name='embedding', 
                           weights=[self.vec.embeddings])(input)
        cnn_network = cnn_embed(lookup, filter_lens, nb_filter, maxlen, word_dim, reg, name='pool')
        lstm_network = lstm_embed(lookup, 128, reg)
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
            
        loss_weights = {}
#         for mod in ['same', 'valid'] :
#             loss_weights[mod + '_' + aspect + '_score'] = 10.0
        
        self.model.compile(optimizer='adam', loss=losses, loss_weights=loss_weights)
        
    def generate_y_batch(self, nb_sample) :
        aspect = self.C['aspect']
        y_batch = {}
        
        y_batch['same_' + aspect + '_score'] = np.ones(nb_sample)
        y_batch['valid_' + aspect + '_score'] = np.ones(nb_sample)
        y_batch['corrupt_' + aspect + '_score'] = np.full(shape=nb_sample, fill_value=-1)
        #y_batch['corrupt_' + aspect + '_score'] = np.ones(nb_sample)
        
        y_batch['corrupt_abstract_score'] = np.full(shape=nb_sample, fill_value=-1)
        y_batch['valid_abstract_score'] = np.ones(nb_sample)
        
        y_batch['valid_' + aspect + '_aspect_score'] = np.ones(nb_sample)
        y_batch['corrupt_' + aspect + '_aspect_score'] = np.full(shape=nb_sample, fill_value=-1)

        for a in self.C['aspect_comp'] :
            for mod in self.modifier :
                y_batch[mod + '_' + a + '_score'] = np.full(shape=nb_sample, fill_value=-1)
            
        return y_batch