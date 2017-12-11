from collections import OrderedDict

import keras.backend as K
from keras.layers import Input, Dense, Lambda, Concatenate, Reshape
from keras.layers import Activation, ActivityRegularization
from keras.layers.merge import Dot, Add, Multiply
from keras.models import Model
from keras.regularizers import l2

from keras.optimizers import Adam

from trainer import Trainer
import numpy as np

def contrastive_loss(y_true, y_pred) :
    return K.mean(K.maximum(0., 1.0 - y_pred), axis=-1)

class CountModel(Trainer) :
    def build_model(self, nb_filter=100, filter_lens=range(1,6), reg=0.000001):
        self.aspects = [str(x) for x in range(0, 4)]

        inputs = {}
        for aspect in self.aspects :
            for mod in ['O'] + self.modifier :
                inputs[(aspect, mod)] = mod + aspect
                        
        vocab_size = self.vec.vocab_size - 2

        input = Input(shape=(vocab_size,), dtype='float32')

        models = OrderedDict()
        gates_models = OrderedDict()
        sum_normalize = Lambda(lambda s : K.l2_normalize(K.sum(s, axis=1), axis=-1))
        normalize = Lambda(lambda s : K.l2_normalize(s, axis=1))

        

        for aspect in self.aspects:
            network = Dense(800, activation='relu', kernel_regularizer=l2(reg))(input)
            network = normalize(network)
            model = Model(input, network)
            model.name = 'pool_' + aspect
            models[aspect] = model  

        I = OrderedDict()
        for input in inputs :
            I[input] =  Input(shape=(vocab_size,), dtype='float32', name=inputs[input])

        C = OrderedDict()
        for aspect in self.aspects :
            for input in inputs :
                if input in [(aspect, 'O'), (aspect, 'S'), (aspect, 'D')] :
                    C[(input, aspect)] = models[aspect](I[input])  

        models_pred = OrderedDict()
        for aspect in self.aspects :
            input_1 = Input(shape=(vocab_size,), dtype='float32', name='A1')
            input_2 = Input(shape=(vocab_size,), dtype='float32', name='A2')
            embed_model = models[aspect]
            embed_1 = embed_model(input_1)
            embed_2 = embed_model(input_2)
            diff = Dot(axes=-1)([embed_1, embed_2])
            output = Activation('linear')(diff)
            model_pred = Model([input_1, input_2], [output])
            model_pred.name = 'pred_'+aspect
            models_pred[aspect] = model_pred

        P = OrderedDict()
        for aspect in self.aspects :
            P[(aspect, 'S')] = models_pred[aspect]([I[(aspect, 'O')], I[(aspect, 'S')]])
            P[(aspect, 'D')] = models_pred[aspect]([I[(aspect, 'O')], I[(aspect, 'D')]])

                    
        D = OrderedDict()
        self.losses = {}

        for aspect in self.aspects :
            name_s = 'O_S'+aspect+'_score'
            
            D[name_s] = Lambda(lambda s : s[0] - s[1], name=name_s)([P[(aspect, 'S')], P[(aspect, 'D')]])

            self.losses[name_s] = contrastive_loss

        self.model = Model(inputs=I.values(), outputs=D.values())
            
        self.model.compile(optimizer=Adam(lr=0.001), loss=self.losses)
        
    def generate_y_batch(self, nb_sample) :
        y_batch = {}

        ones = []
        zeros = []
        for aspect in self.aspects :
            name_s = 'O_S'+aspect+'_score'
            name_d = 'O_D'+aspect+'_score'

            ones.append(name_s)
            zeros.append(name_d)


        zeros.append('gate_reg')

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

    def calc_sim(self, x, y, aspect) :
        layer = self.model.get_layer('pred_' + aspect)
        inputs = layer.inputs + [K.learning_phase()]
        outputs = layer.outputs
        sim_function = K.function(inputs, outputs)
        sim = sim_function([x, y, 0])[0]
        return sim


        
