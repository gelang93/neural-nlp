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
    return K.mean((1 - y_true) * K.maximum(0., y_pred) + (y_true) * K.maximum(0., 1. - y_pred), axis=-1)

class GatedCNNModel(Trainer) :
    def build_model(self, nb_filter=100, filter_lens=range(1,6), reg=0.00001):
        self.aspects = ['P', 'I', 'O']

        inputs = {}
        for aspect in self.aspects + ['A'] :
            for mod in self.modifier :
                inputs[(aspect, mod)] = mod + aspect
                        
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
        
        models = OrderedDict()
        gates_models = OrderedDict()

        sum_normalize = Lambda(lambda s : K.l2_normalize(K.sum(s, axis=1),axis=-1))
        normalize = Lambda(lambda s : K.l2_normalize(s, axis=1))

        for aspect in self.aspects + ['E']:
            network1, gates1 = gated_cnn(lookup, 1, 200, reg)
            network1 = mwp(network1)

            network2, gates2 = gated_cnn(network1, 3, 200, reg)
            network2 = mwp(network2)

            network2 = Add()([network1, network2])
                   
            network3, gates3 = gated_cnn(network2, 5, 200, reg)
            network3 = mwp(network3)

            network3 = Add()([network1, network2, network3])

            #network4, gates4 = gated_cnn(network3, 7, 200, reg)
            gates = mwp(gates3)
            
            network3 = Multiply()([network3, gates])
            # network3 = CuDNNGRU(128, kernel_regularizer=l2(reg), return_sequences=True)(lookup)
            # gates = TimeDistributed(Dense(1, activation='sigmoid', kernel_regularizer=l2(reg), activity_regularizer=l1(reg)))(network3)
            # gates = mwp(gates)
            # network3 = Multiply()([network3, gates])
            network3 = mwp(network3)

            network3 = sum_normalize(network3)
            network = network3

            gates_network = gates          
            gates_network = normalize(gates_network)

            model = Model(input, network)
            model.name = 'pool_' + aspect
            models[aspect] = model       

            gate_model = Model(input, gates_network)     
            gates_models[aspect] = gate_model

        D = OrderedDict()            
        I = OrderedDict()
        for input in inputs :
            I[input] = Input(shape=[maxlen], dtype='float32', name=inputs[input])
                        
        C = OrderedDict()
        for aspect in self.aspects + ['E'] :
            for input in inputs :
                C[(input, aspect)] = models[aspect](I[input])     
        
        G1 = OrderedDict()
        for aspect in self.aspects + ['E']:
            G1[aspect] = gates_models[aspect](I[('A', 'S')])

        gate_concat = Concatenate()(G1.values())
        gate_dot = Dot(axes=1)([gate_concat, gate_concat])
        gate_reg = Lambda(lambda s : 0.1 * K.sum(K.sum(K.square(s - K.eye(4)), axis=-1), 
                                            axis=-1, keepdims=True), 
                            name='gate_reg')(gate_dot)
                    
        #D['gate_reg'] = gate_reg
        
        self.losses = {}
        #self.losses['gate_reg'] = lambda y_true, y_pred : K.mean(y_pred, axis=-1)
        self.loss_weights = {}

        for aspect in self.aspects :
            embed_1 = C[(('A', 'S'), aspect)]
            for mod in self.modifier :
                embed_2 = C[((aspect, mod), aspect)]
                name = 'SA' + aspect + '_' + mod + aspect + aspect + '_score'
                D[name] = Dot(axes=1, name=name)([embed_1, embed_2])
                self.losses[name] = contrastive_loss
            for aspect_comp in (set(self.aspects) - set([aspect])) :
                embed_2 = C[((aspect_comp, 'S'), aspect)]
                name = 'SA' + aspect + '_S' + aspect_comp + aspect + '_score' 
                D[name] = Dot(axes=1, name=name)([embed_1, embed_2])
                self.losses[name] = contrastive_loss

                # embed_3 = C[((aspect_comp, 'V'), aspect)]
                # name = 'SA' + aspect + '_V' + aspect_comp + aspect + '_score'
                # D[name] = Dot(axes=1, name=name)([embed_1, embed_3])
                # self.losses[name] = contrastive_loss
            
            for mod in ['V', 'C'] :
                embed_2 = C[(('A',mod), aspect)]
                name = 'SA' + aspect + '_' + mod + 'A' + aspect + '_score'
                D[name] = Dot(axes=1, name=name)([embed_1, embed_2])
                self.losses[name] = contrastive_loss
                   
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
                ones.append('SA' + aspect + '_V' + aspect_comp + aspect + '_score')
                
            name = 'SA' + aspect + '_' + 'VA' + aspect + '_score'
            ones.append(name)
            name = 'SA' + aspect + '_' + 'CA' + aspect + '_score'
            neg_ones.append(name)
            
            name = 'SAE_S' + aspect + 'E_score'
            neg_ones.append(name)
            
        ones.append('SAE_VAE_score')
        neg_ones.append('SAE_CAE_score')
        neg_ones.append('gate_reg')
                        
        y_batch = {}
        for loss in ones :
            y_batch[loss] = np.ones(nb_sample)
            
        for loss in neg_ones :
            y_batch[loss] = np.full(shape=nb_sample, fill_value=0)
            
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
