from collections import OrderedDict

import keras.backend as K
from keras.layers import Input, Embedding, Dropout, Lambda, Concatenate, RepeatVector, Permute
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, merge
from keras.layers import Activation, Lambda, ActivityRegularization
from keras.layers.merge import Dot
from keras.models import Model
from keras.regularizers import l2

from trainer import Trainer
from support import cnn_embed, lstm_embed, Element_wise_weighting, cnn_embed_reshape

import numpy as np

class OneCNNModel(Trainer) :
    def build_model(self, nb_filter=100, filter_lens=range(1,6), reg=0.0001):
        self.aspects = ['P', 'I', 'O']
        self.modifier = ['S', 'V', 'C']
        inputs = {}
        for aspect in self.aspects + ['A'] :
            for mod in self.modifier :
                inputs[(aspect, mod)] = mod + aspect
                        
        maxlen = self.vec.maxlen
        word_dim, vocab_size = self.vec.word_dim, self.vec.vocab_size

        input = Input(shape=[maxlen], dtype='int32')
        lookup = Embedding(output_dim=word_dim, 
                           input_dim=vocab_size, 
                           name='embedding', 
                           weights=[self.vec.embeddings])(input)
        
        #cnn_network = cnn_embed_reshape(lookup, filter_lens, nb_filter*4, 4, maxlen, reg)
        cnn_network, convs = cnn_embed(lookup, filter_lens, nb_filter * len(self.aspects), maxlen, reg)
        weighted = Element_wise_weighting(nb_aspects=len(self.aspects))(cnn_network)

        models = OrderedDict()
        for i, aspect in enumerate(self.aspects) :
            masked_aspect = Lambda(lambda s : s[:,:,i])(weighted)
            model_aspect = Model(input, masked_aspect)
            models[aspect] = model_aspect
            models[aspect].name = 'pool_' + aspect

        I = OrderedDict()
        for input in inputs :
            I[input] = Input(shape=[maxlen], dtype='int32', name=inputs[input])
                        
        C = OrderedDict()
        for aspect in self.aspects :
            for input in inputs :
                C[(input, aspect)] = models[aspect](I[input])     
                    
        D = OrderedDict()
        self.losses = {}

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
            
        # for aspect in self.aspects :
        #     embed_1 = C[(('A', 'S'), 'E')]
        #     embed_2 = C[((aspect, 'S'), 'E')]
        #     name = 'SAE_S' + aspect + 'E_score'
        #     D[name] = Dot(axes=1, name=name)([embed_1, embed_2])
        #     self.losses[name] ='hinge'
            
        # for mod in ['V', 'C'] :
        #     embed_1 = C[(('A','S'),'E')]
        #     embed_2 = C[(('A', mod), 'E')]
        #     name = 'SAE_' + mod + 'AE_score'
        #     D[name] = Dot(axes=1, name=name)([embed_1, embed_2])
        #     self.losses[name] = 'hinge'
                    
        self.model = Model(inputs=I.values(), outputs=D.values())
            
        self.model.compile(optimizer='adam', loss=self.losses)
        
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
            
        from itertools import combinations
        aspect_comb = list(combinations(self.aspects, 2))
        
        for comb in aspect_comb :
            name = 'SA' + comb[0] + '_SA' + comb[1] + '_score'
            neg_ones.append(name)
            
        ones.append('pool_score')
            
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
        