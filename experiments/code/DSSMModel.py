from collections import OrderedDict

import keras.backend as K
from keras.layers import Input, Embedding, Dropout, Dense, LSTM, merge, Lambda, Concatenate
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, merge
from keras.layers import Activation, Lambda, ActivityRegularization, TimeDistributed, CuDNNGRU
from keras.layers.merge import Dot, Multiply, Add
from keras.models import Model
from keras.regularizers import l2, l1
from gcnn import GCNN
from trainer import Trainer
from support import cnn_embed, lstm_embed, gated_cnn, gated_cnn_joint

import numpy as np

def contrastive_loss(y_true, y_pred) :
    return K.mean(K.maximum(0., 1. - y_pred), axis=-1)

def contrastive_loss(y_true, y_pred) :
    return K.mean(-K.log(y_pred[:, 0]), axis=-1)

class DSSMModel(Trainer) :
    def build_model(self, nb_filter=100, filter_lens=range(1,6), reg=0.00001):
        self.aspects = ['P', 'I', 'O']

        inputs = {}
        for mod in self.modifier :
            inputs[('A', mod)] = mod + 'A'
                        
        maxlen = self.vec.maxlen
        word_dim, vocab_size = self.vec.word_dim, self.vec.vocab_size

        input = Input(shape=[maxlen], dtype='float32')        
        padding = Lambda(lambda s : K.expand_dims(K.clip(s, 0, 1)))(input)

        def mwp(seq) :
            return Multiply()([seq, padding])

        lookup = Embedding(output_dim=word_dim, 
                           input_dim=vocab_size, 
                           name='embedding', 
                           weights=[self.vec.embeddings])(input)
        lookup = mwp(lookup)    
        cnn, conv = cnn_embed(lookup, filter_lens, 300, maxlen, reg)
        dense = Dense(256, activation='tanh')(cnn)
        normalize = Lambda(lambda s : K.l2_normalize(s, axis=1))(dense)

        self.model_dssm = Model(input, normalize)

        D = OrderedDict()            
        I = OrderedDict()
        for input in inputs :
            I[input] = Input(shape=[maxlen], dtype='float32', name=inputs[input])
                        
        C = OrderedDict()
        for input in inputs :
            C[input] = self.model_dssm(I[input])     
        
        self.losses = {}
        self.loss_weights = {}
        
        embed_1 = C[('A', 'S')]
        embed_2 = C[('A', 'V')]
        embed_3 = C[('A', 'C')]
        d1 = Dot(axes=1)([embed_1, embed_2])
        d2 = Dot(axes=1)([embed_1, embed_3])
        D['diff'] = Lambda(lambda s : K.softmax(K.concatenate([s[0], s[1]], axis=-1)), name='diff')([d1, d2])

        self.losses['diff'] = contrastive_loss

        self.model = Model(inputs=I.values(), outputs=D.values())
            
        self.model.compile(optimizer='adam', loss=self.losses, loss_weights=self.loss_weights)
        
    def generate_y_batch(self, nb_sample) :
        ones = []
        neg_ones = []
        ones.append('diff')
                        
        y_batch = {}
        for loss in ones :
            y_batch[loss] = np.ones(nb_sample)
            
        for loss in neg_ones :
            y_batch[loss] = np.full(shape=nb_sample, fill_value=0)
            
        return y_batch
    
    def construct_evaluation_model(self, model, aspect_specific=False) :
        inputs = []
        outputs = []
        model_aspect = self.model_dssm
        inputs += model_aspect.inputs
        outputs += model_aspect.outputs

        inputs += [K.learning_phase()]
        self.aspect_evaluation_models = {}
        for i, aspect in enumerate(self.aspects) :
            self.aspect_evaluation_models[aspect] = K.function(inputs, outputs)
        
        self.evaluation_model = K.function(inputs, outputs)
        if aspect_specific :
            return self.evaluation_model, self.aspect_evaluation_models
        return self.evaluation_model
