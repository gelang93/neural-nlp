from collections import OrderedDict

import keras.backend as K
from keras.layers import Input, Embedding, Lambda
from keras.layers.merge import Dot, Multiply, Add
from keras.models import Model
from trainer import Trainer
from support import gated_cnn

import numpy as np

def contrastive_loss(y_true, y_pred) :
    return K.mean((1 - y_true) * K.maximum(0., y_pred) + (y_true) * K.maximum(0., 1. - y_pred), axis=-1)

class GatedCNNModel(Trainer) :
    def build_model(self, reg=0.00001):
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

        network1, gates1 = gated_cnn(lookup, 3, 200, reg)
        network1 = mwp(network1)
        
        models = OrderedDict()

        sum_normalize = Lambda(lambda s : K.l2_normalize(K.sum(s, axis=1),axis=-1))

        for aspect in self.aspects + ['E']:
            # network1, gates1 = gated_cnn(lookup, 1, 200, reg)
            # network1 = mwp(network1)

            network2, gates2 = gated_cnn(network1, 3, 200, reg)
            network2 = mwp(network2)

            network2 = Add()([network1, network2])
                   
            network3, gates3 = gated_cnn(network2, 5, 200, reg)
            network3 = mwp(network3)

            network3 = Add()([network1, network2, network3])

            gates = mwp(gates3)
            
            network3 = Multiply()([network3, gates])
            network3 = mwp(network3)

            network3 = sum_normalize(network3)
            network = network3

            model = Model(input, network)
            model.name = 'pool_' + aspect
            models[aspect] = model       

        D = OrderedDict()            
        I = OrderedDict()
        for input in inputs :
            I[input] = Input(shape=[maxlen], dtype='float32', name=inputs[input])
                        
        C = OrderedDict()
        for aspect in self.aspects + ['E'] :
            for input in inputs :
                C[(input, aspect)] = models[aspect](I[input])     
        
        self.losses = {}
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
            
            for mod in ['V', 'C'] :
                embed_2 = C[(('A',mod), aspect)]
                name = 'SA' + aspect + '_' + mod + 'A' + aspect + '_score'
                D[name] = Dot(axes=1, name=name)([embed_1, embed_2])
                self.losses[name] = contrastive_loss
                   
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

        output = K.concatenate(outputs, axis=-1)
        self.evaluation_model = K.function(inputs,[output])

        self.aspect_evaluation_models = {}
        for aspect in self.aspects :
            self.aspect_evaluation_models[aspect] = K.function(model.get_layer('pool_'+aspect).inputs + [K.learning_phase()], 
                                                                [model.get_layer('pool_'+aspect).outputs[0]])

        
        if aspect_specific :
            return self.evaluation_model, self.aspect_evaluation_models
        return self.evaluation_model
