from collections import OrderedDict

import keras.backend as K
from keras.layers import Input, Embedding, Dropout, Dense, LSTM, merge, Lambda, Concatenate
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, merge
from keras.layers import Activation, Lambda, ActivityRegularization
from keras.layers.merge import Dot, Add
from keras.models import Model
from keras.regularizers import l2

from trainer import Trainer
from support import cnn_embed, lstm_embed, norm

import numpy as np

class CrazyCNNModel(Trainer) :
    def build_model(self, nb_filter=300, filter_lens=range(1,6), reg=0.0001):
        self.aspects = ['P', 'I', 'O']

        inputs = {}
        for aspect in self.aspects + ['A']:
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
        convolves = OrderedDict()
        for aspect in self.aspects + ['E'] :
            cnn_network, convolutions = cnn_embed(lookup, filter_lens, nb_filter, maxlen, word_dim, reg, name='pool_'+aspect)
            model = Model(input, cnn_network)
            model.name = 'pool_' + aspect
            models[aspect] = model    
            
            conv = Model(input, convolutions)
            conv.name = 'conv_' + aspect
            convolves[aspect] = conv
                    
        I = OrderedDict()
        for input in inputs :
            I[input] = Input(shape=[maxlen], dtype='int32', name=inputs[input])
                        
        C = OrderedDict()
        for aspect in self.aspects + ['E'] :
            for input in inputs :
                C[(input, aspect)] = models[aspect](I[input])   
                
        F = OrderedDict()
        for aspect in self.aspects + ['E']:
            for input in inputs :
                F[(input, aspect)] = convolves[aspect](I[input])
                    
        D = OrderedDict()
        self.losses = {}
        self.loss_weights = {}

        for aspect in self.aspects :
            embed_1 = C[(('A', 'S'), aspect)]
            for mod in self.modifier :
                embed_2 = C[((aspect, mod), aspect)]
                name = 'SA' + aspect + '_' + mod + aspect + aspect + '_score'
                D[(('A', 'S'), (aspect, mod))] = Dot(axes=1, name=name)([embed_1, embed_2])
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

#         aspect_filters = [F[(('A','S'),aspect)] for aspect in self.aspects + ['E']]
#         name = 'aspect_filter_dot'
        
#         sum_layer = Lambda(lambda s : K.sum(s, axis=-1, keepdims=True))
#         aspect_filter_sums = [sum_layer(aspect_filter) for aspect_filter in aspect_filters]
#         d = Lambda(lambda s : s[0]*s[1]*s[2])(aspect_filter_sums)
#         D[name] = Lambda(lambda s : K.mean(s, axis=1), name=name)(d)
#         self.losses[name] = lambda y_true, y_pred : K.mean(y_pred)
        # import pdb
        # pdb.set_trace()
        
        
            
#         for aspect, afs in zip(self.aspects + ['E'], aspect_filters) :
#             l2_layer = Lambda(lambda s : 1/norm(s), name=name+'_l2_' + aspect)(afs)
#             D[name+'l2_layer_'+aspect] = l2_layer
#             self.losses[name + '_l2_' + aspect] = lambda y_true, y_pred : K.mean(y_pred)
        
        # add_norm_layer = Add()(aspect_filter_sums)
        # l2_layer = Lambda(lambda s : 1/norm(s, keepdims=False), name=name+'_l2')(add_norm_layer)
        # D[name+'l2_layer'] = l2_layer
        # self.losses[name + '_l2'] = lambda y_true, y_pred : K.mean(y_pred)
        
        
            #self.loss_weights[name] = 5.0
                    
        self.model = Model(inputs=I.values(), outputs=D.values())
            
        self.model.compile(optimizer='adam', loss=self.losses)#, loss_weights=self.loss_weights)
        
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
            
        ones.append('aspect_filter_dot')
        ones.append('aspect_filter_dot_l2')
        for aspect in self.aspects + ['E'] :
            ones.append('aspect_filter_dot_l2_'+aspect)
                    
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
        