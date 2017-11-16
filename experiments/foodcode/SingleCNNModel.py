from collections import OrderedDict

import keras.backend as K
from keras.layers import Input, Embedding, Dropout, Dense, LSTM, merge, Lambda, Concatenate
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, merge
from keras.layers import Activation, Lambda, ActivityRegularization
from keras.layers.merge import Dot
from keras.models import Model
from keras.regularizers import l2

from keras.optimizers import Adam

from trainer import Trainer
from support import cnn_embed, Bilinear

import numpy as np

def contrastive_loss(y_true, y_pred) :
    return K.mean((1 - y_true) * y_pred + (y_true) * K.maximum(0., 1. - y_pred), axis=-1)

#Dot = lambda axes, name : Lambda(lambda s : K.sum(K.square(K.l2_normalize(s[0] - s[1]), axis=-1, keepdims=True)), name=name)

class SingleCNNModel(Trainer) :
    def build_model(self, nb_filter=300, filter_lens=range(1,6), reg=0.0001):
        self.aspects = [str(x) for x in range(0, 4)]

        inputs = {}
        inputs['O'] = 'O'
        for aspect in self.aspects :
            for mod in self.modifier :
                inputs[(aspect, mod)] = mod + aspect
                        
        maxlen = self.vec.maxlen
        word_dim, vocab_size = self.vec.word_dim, self.vec.vocab_size

        input = Input(shape=[maxlen], dtype='int32')
        lookup = Embedding(output_dim=word_dim, 
                           input_dim=vocab_size, 
                           name='embedding',
                           weights=[self.vec.embeddings])(input)
        
        models = OrderedDict()
        for aspect in self.aspects:
            cnn_network, convs = cnn_embed(lookup, filter_lens, nb_filter, maxlen, word_dim, reg, name='pool_'+aspect)
            model = Model(input, cnn_network)
            model.name = 'pool_' + aspect
            models[aspect] = model            
                    
        I = OrderedDict()
        for input in inputs :
            I[input] = Input(shape=[maxlen], dtype='int32', name=inputs[input])
                        
        C = OrderedDict()
        for aspect in self.aspects :
            for input in inputs :
                if input in ['O', (aspect, 'S'), (aspect, 'D')] :
                    C[(input, aspect)] = models[aspect](I[input])     
                    
        D = OrderedDict()
        self.losses = {}
        
        for aspect in self.aspects :
            embed_1 = C[('O', aspect)]
            embed_s = C[((aspect, 'S'), aspect)]
            embed_d = C[((aspect, 'D'), aspect)]

            #embed_2 = Concatenate()([embed_1, C[((aspect, 'S'), aspect)]])
            #embed_3 = Concatenate()([embed_1, C[((aspect, 'D'), aspect)]])

            name_s = 'O_S'+aspect+'_score'
            name_d = 'O_D'+aspect+'_score'

            B = Dense(1, name='B'+aspect)
            Bs = Activation('sigmoid', name=name_s)(B(Concatenate()([embed_1, embed_s])))
            Bd = Activation('sigmoid', name=name_d)(B(Concatenate()([embed_1, embed_d])))
            
            #D[name_s] = Dot(axes=1, normalize='l2', name=name_s)([embed_1, embed_s])
            #D[name_d] = Dot(axes=1, normalize='l2', name=name_d)([embed_1, embed_d])
            
            D[name_s] = Bs
            D[name_d] = Bd

            #self.losses[name_s] = contrastive_loss
            #self.losses[name_d] = contrastive_loss

            self.losses[name_s] = 'binary_crossentropy'
            self.losses[name_d] = 'binary_crossentropy'
                   
        self.model = Model(inputs=I.values(), outputs=D.values())
            
        self.model.compile(optimizer='adam', loss=self.losses)
        
    def generate_y_batch(self, nb_sample) :
        y_batch = {}

        ones = []
        zeros = []
        for aspect in self.aspects :
            name_s = 'O_S'+aspect+'_score'
            name_d = 'O_D'+aspect+'_score'
            ones.append(name_s)
            zeros.append(name_d)

        for loss in ones :
            y_batch[loss] = np.ones(nb_sample)
            
        for loss in zeros :
            y_batch[loss] = np.zeros(nb_sample)
            
        return y_batch

    def construct_evaluation_model(self, model, aspect_specific=False) :
        inputs = []
        outputs = []
        for aspect in self.aspects :
            model_aspect = model.get_layer('pool_' + aspect)
            inputs += model_aspect.inputs
            outputs += model_aspect.outputs
        inputs = inputs[:1]
        inputs += [K.learning_phase()]
        self.aspect_evaluation_models = {}
        for i, aspect in enumerate(self.aspects) :
            self.aspect_evaluation_models[aspect] = K.function(inputs, [outputs[i]])
        output = K.concatenate(outputs, axis=-1)
        self.evaluation_model = K.function(inputs,[output])
        if aspect_specific :
            return self.evaluation_model, self.aspect_evaluation_models
        return self.evaluation_model

    # def construct_evaluation_model(self, model, aspect_specific=False) :
    #     inputs = []
    #     outputs = []


        