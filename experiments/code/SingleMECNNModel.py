from collections import OrderedDict

import keras.backend as K
from keras.layers import Input, Embedding, Dropout, Dense, LSTM, merge, Lambda, Concatenate
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, merge
from keras.layers import Activation, Lambda, ActivityRegularization
from keras.layers.merge import Dot
from keras.models import Model
from keras.regularizers import l2

from trainer import Trainer
from support import cnn_embed_me

import numpy as np

class SingleMECNNModel(Trainer) :
    def build_model(self, nb_filter=300, filter_lens=range(1,6), reg=0.0001):
        self.aspects = ['P', 'I', 'O']

        inputs = {}
        for aspect in self.aspects + ['A'] :
            for mod in self.modifier :
                inputs[(aspect, mod)] = mod + aspect
                        
        maxlen = self.vec.maxlen
        word_dim, vocab_size = self.vec.word_dim, self.vec.vocab_size

        input = Input(shape=[maxlen], dtype='int32')
        embedding_layer = Embedding(output_dim=word_dim, 
                           input_dim=vocab_size, 
                           name='embedding', 
                           weights=[self.vec.embeddings])
        lookup = embedding_layer(input)
        
        input_aspect = Input(shape=[200], dtype='int32')#Lambda(lambda s : s[:, -200:])(input)
        lookup_aspect = embedding_layer(input_aspect)
        
        models = OrderedDict()
        for aspect in self.aspects + ['E']:
            net_abs, net_asp = cnn_embed_me(lookup, lookup_aspect, filter_lens, nb_filter, maxlen, 200, reg)
            model_abs = Model(input, net_abs)
            model_asp = Model(input_aspect, net_asp)
            model_abs.name = 'pool_' + aspect
            model_asp.name = 'pool_' + aspect + '_aspect'
            models[(aspect, 'A')] = model_abs
            for aspect_in in self.aspects :
                models[(aspect, aspect_in)] = model_asp
                    
        I = OrderedDict()
        for input in inputs :
            if input[0] in self.aspects :
                I[input] = Input(shape=[200], dtype='int32', name=inputs[input])
            else :
                I[input] = Input(shape=[maxlen], dtype='int32', name=inputs[input])
                        
        C = OrderedDict()
        for aspect in self.aspects + ['E'] :
            for input in inputs :
                C[(input, aspect)] = models[(aspect, input[0])](I[input])     
                    
        D = OrderedDict()
        self.losses = {}
        
        self.loss_weights = {}

        for aspect in self.aspects :
            embed_1 = C[(('A', 'S'), aspect)]
            for mod in self.modifier :
                embed_2 = C[((aspect, mod), aspect)]
                name = 'SA' + aspect + '_' + mod + aspect + aspect + '_score'
                D[name] = Dot(axes=1, name=name)([embed_1, embed_2])
                self.losses[name] = 'hinge'
            for aspect_comp in (set(self.aspects) - set([aspect])) :
                embed_2 = C[((aspect_comp, 'S'), aspect)]
                name = 'SA' + aspect + '_S' + aspect_comp + aspect + '_score' 
                D[name] = Dot(axes=1, name=name)([embed_1, embed_2])
                self.losses[name] = 'hinge'
            
            for mod in ['V', 'C'] :
                embed_2 = C[(('A',mod), aspect)]
                name = 'SA' + aspect + '_' + mod + 'A' + aspect + '_score'
                D[name] = Dot(axes=1, name=name)([embed_1, embed_2])
                self.losses[name] = 'hinge'  
                
            # for mod in ['S', 'V'] :
            #     name = 'SA' + aspect + '_' + mod + aspect + aspect + '_score'
            #     self.loss_weights[name] = 5.0
                
            #self.loss_weights['SA' + aspect + '_' + 'VA' + aspect + '_score'] = 5.0
            
        for aspect in self.aspects :
            embed_1 = C[(('A', 'S'), 'E')]
            embed_2 = C[((aspect, 'S'), 'E')]
            name = 'SAE_S' + aspect + 'E_score'
            D[name] = Dot(axes=1, name=name)([embed_1, embed_2])
            self.losses[name] ='hinge'
            
        for mod in ['V', 'C'] :
            embed_1 = C[(('A','S'),'E')]
            embed_2 = C[(('A', mod), 'E')]
            name = 'SAE_' + mod + 'AE_score'
            D[name] = Dot(axes=1, name=name)([embed_1, embed_2])
            self.losses[name] = 'hinge'
              
        #self.loss_weights['SAE_VAE_score'] = 5.0
                   
        print self.loss_weights
        self.model = Model(inputs=I.values(), outputs=D.values())
            
        self.model.compile(optimizer='adam', loss=self.losses, loss_weights=self.loss_weights)
        
    def generate_y_batch(self, nb_sample) :
        ones = []
        neg_ones = []
        for aspect in self.aspects :
            for mod in ['S', 'V'] :
                name = 'SA' + aspect + '_' + mod + aspect + aspect + '_score'
                ones.append(name)
            mod = 'C'
            name = 'SA' + aspect + '_' + mod + aspect + aspect + '_score'
            neg_ones.append(name)
            for aspect_comp in (set(self.aspects) - set([aspect])) :
                name = 'SA' + aspect + '_S' + aspect_comp + aspect + '_score' 
                neg_ones.append(name)
                
            name = 'SA' + aspect + '_' + 'VA' + aspect + '_score'
            ones.append(name)
            name = 'SA' + aspect + '_' + 'CA' + aspect + '_score'
            neg_ones.append(name)
            
            name = 'SAE_S' + aspect + 'E_score'
            neg_ones.append(name)
            
        ones.append('SAE_VAE_score')
        neg_ones.append('SAE_CAE_score')
                        
        y_batch = {}
        for loss in ones :
            y_batch[loss] = np.ones(nb_sample)
            
        for loss in neg_ones :
            y_batch[loss] = np.full(shape=nb_sample, fill_value=-1)
            
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