from collections import OrderedDict

import keras.backend as K
from keras.layers import Input, Embedding, Lambda
from keras.layers import Activation
from keras.layers.merge import Dot, Add, Multiply
from keras.models import Model

from keras.optimizers import Adam

from trainer import Trainer
from support import gated_cnn

import numpy as np

def contrastive_loss(y_true, y_pred) :
    return K.mean(K.maximum(0., 1.0 - y_pred), axis=-1)

class Gated2CNNModel(Trainer) :
    def build_model(self, reg=0.000001):
        self.aspects = [str(x) for x in range(0, 4)]

        inputs = {}
        for aspect in self.aspects :
            for mod in ['O'] + self.modifier :
                inputs[(aspect, mod)] = mod + aspect
                        
        maxlen = self.vec.maxlen
        word_dim, vocab_size = self.vec.word_dim, self.vec.vocab_size

        input = Input(shape=[maxlen], dtype='int32')
        padding = Lambda(lambda s : K.expand_dims(K.cast(K.clip(s, 0, 1), 'float32')))(input)

        def mwp(seq) :
            return Multiply()([seq, padding])        

        lookup = Embedding(output_dim=word_dim, 
                           input_dim=vocab_size, 
                           name='embedding',
                           weights=[self.vec.embeddings])(input)
        lookup = mwp(lookup)

        network1, gates1 = gated_cnn(lookup, 1, 200, reg)
        network1 = mwp(network1)

        sum_normalize = Lambda(lambda s : K.l2_normalize(K.sum(s, axis=1), axis=-1))

        models = OrderedDict()

        for aspect in self.aspects:
            

            network2, gates2 = gated_cnn(network1, 3, 200, reg)
            network2 = mwp(network2)
            network2 = Add()([network1, network2])
            
            network3, gates3 = gated_cnn(network2, 5, 200, reg)
            network3 = mwp(network3)
            network3 = Add()([network1, network2, network3])

            gates1, gates2, gates3 = mwp(gates1), mwp(gates2), mwp(gates3)
            network3 = Multiply()([network3, gates3])

            network3 = mwp(network3)

            network3_sum = sum_normalize(network3)

            aspect_network = network3_sum
            
            model = Model(input, aspect_network)
            model.name = 'pool_' + aspect
            models[aspect] = model  

        I = OrderedDict()
        for input in inputs :
            I[input] = Input(shape=[maxlen], dtype='int32', name=inputs[input]) 


        models_pred = OrderedDict()
        for aspect in self.aspects :
            input_1 = Input(shape=[maxlen], dtype='int32', name='A1')
            input_2 = Input(shape=[maxlen], dtype='int32', name='A2')
            input_3 = Input(shape=[maxlen], dtype='int32', name='A3')
            embed_model = models[aspect]
            embed_1 = embed_model(input_1)
            embed_2 = embed_model(input_2)
            embed_3 = embed_model(input_3)
            diff1 = Dot(axes=-1)([embed_1, embed_2])
            diff2 = Dot(axes=-1)([embed_1, embed_3])
            output = Lambda(lambda s : s[0] - s[1])([diff1, diff2])
            model_pred = Model([input_1, input_2, input_3], [output])
            model_pred.name = 'pred_'+aspect
            models_pred[aspect] = model_pred

        P = OrderedDict()
        for aspect in self.aspects :
            P[aspect] = models_pred[aspect]([I[(aspect, 'O')], I[(aspect, 'S')], I[(aspect, 'D')]])
                    
        D = OrderedDict()
        self.losses = {}

        for aspect in self.aspects :
            name_s = 'O_S'+aspect+'_score'
            output_1 = P[aspect]
            D[name_s] = Activation('linear', name=name_s)(output_1)
            self.losses[name_s] = contrastive_loss
            

        self.model = Model(inputs=list(I.values()), outputs=list(D.values()))
            
        self.model.compile(optimizer=Adam(lr=0.001), loss=self.losses)
        
    def generate_y_batch(self, nb_sample) :
        y_batch = {}

        ones = []
        zeros = []
        for aspect in self.aspects :
            name_s = 'O_S'+aspect+'_score'
            ones.append(name_s)
        

        for loss in ones :
            y_batch[loss] = np.ones(nb_sample)
            
        for loss in zeros :
            y_batch[loss] = np.zeros(nb_sample)

        for loss in zeros :
            y_batch[loss] = np.full(nb_sample, 0)
            
        return y_batch

    def construct_evaluation_model(self, model, aspect_specific=False) :
        inputs = []
        outputs = []
        for aspect in self.aspects :
            model_aspect = model.get_layer('pred_'+aspect).get_layer('pool_' + aspect)
            inputs += model_aspect.inputs
            outputs += [model_aspect.outputs[0]]
        inputs = inputs[:1]
        inputs += [K.learning_phase()]
        output = K.concatenate(outputs, axis=-1)
        self.evaluation_model = K.function(inputs,[output])

        self.aspect_evaluation_models = {}
        for i, aspect in enumerate(self.aspects) :
            model_aspect = model.get_layer('pred_'+aspect).get_layer('pool_' + aspect)
            input = model_aspect.inputs + [K.learning_phase()]
            output = model_aspect.outputs
            self.aspect_evaluation_models[aspect] = K.function(input, output)
        

        if aspect_specific :
            return self.evaluation_model, self.aspect_evaluation_models
        return self.evaluation_model

    def calc_sim(self, x, y, aspect) :
        layer = self.model.get_layer('pred_' + aspect)
        inputs = layer.inputs + [K.learning_phase()]
        outputs = layer.outputs
        sim_function = K.function(inputs, outputs)
        sim = sim_function([x, y, 0])[0]
        return sim


        
