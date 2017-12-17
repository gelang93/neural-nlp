from collections import OrderedDict

import keras.backend as K
from keras.layers import Input, Dense, Lambda, Concatenate, Reshape
from keras.layers import Activation, ActivityRegularization
from keras.layers.merge import Dot, Add, Multiply
from keras.models import Model
from keras.regularizers import l2
from keras.backend import random_normal
from keras.optimizers import Adam

from trainer import Trainer
import numpy as np

def contrastive_loss(y_true, y_pred) :
    return K.mean(K.maximum(0., 1.0 - y_pred), axis=-1)

def sample_norm(args):
    '''reparameterized sampling from normal distribution'''
    mu, log_var = args
    epsilon = K.random_normal(shape=(K.shape(mu)[0], 200))
    return mu + K.exp(0.5 * log_var) * epsilon

def log_softmax(x, axis=None):
    x0 = x - K.max(x, axis=axis, keepdims=True)
    log_sum_exp_x0 = K.log(K.sum(K.exp(x0), axis=axis, keepdims=True))
    return x0 - log_sum_exp_x0

def cross_ent_loss(s): 
    return - K.sum(s[0] * log_softmax(s[1], axis=-1), axis=-1, keepdims=True) 

def kld_gauss(s) :
    m1, s1, m2, s2 = s
    kl = s2 - s1 + 0.5 * (-K.exp(2*s2) + K.exp(2 * s1) + (m1 - m2) ** 2) / K.exp(2 * s2)
    return K.sum(kl, axis=-1, keepdims=True)
    

class NVDMModel(Trainer) :
    def build_model(self, nb_filter=100, filter_lens=range(1,6), reg=0.000001):
        self.aspects = [str(x) for x in range(0, 4)]

        inputs = {}
        for aspect in self.aspects :
            for mod in ['O'] + self.modifier :
                inputs[(aspect, mod)] = mod + aspect
                        
        vocab_size = self.vec.vocab_size - 2

        input = Input(shape=(vocab_size,), dtype='float32')

        models = OrderedDict()

        kld_loss = Lambda(lambda s : - 0.5 * K.sum(1 + 2*s[1] - K.square(s[0]) - K.exp(2*s[1]), axis=-1, keepdims=True))
        cross_ent = Lambda(cross_ent_loss)
        
        for aspect in self.aspects:
            h = Dense(500, activation='tanh')(input)
            mu = Dense(200)(h)
            log_sigma = Dense(200)(h)

            f = Dense(vocab_size)

            z = Lambda(sample_norm, name='z'+aspect)([mu, log_sigma])
            r = f(z)
            cross = cross_ent([input, r])
            kld = kld_loss([mu, log_sigma])
            vae_term = Add()([cross, kld])
            model = Model(input, [mu, log_sigma, vae_term])
            model.name = 'pool_' + aspect
            models[aspect] = model  

        I = OrderedDict()
        for input in inputs :
            I[input] =  Input(shape=(vocab_size,), dtype='float32', name=inputs[input]) 

        models_pred = OrderedDict()
        for aspect in self.aspects :
            input_1 = Input(shape=(vocab_size,), dtype='float32', name='A1')
            input_2 = Input(shape=(vocab_size,), dtype='float32', name='A2')
            input_3 = Input(shape=(vocab_size,), dtype='float32', name='A3')

            embed_model = models[aspect]
            embed_1, sigma_1, vae_1 = embed_model(input_1)
            embed_2, sigma_2, vae_2 = embed_model(input_2)
            embed_3, sigma_3, vae_3 = embed_model(input_3)

            diff1 = Dot(axes=-1, normalize='l2')([embed_1, embed_2])
            diff2 = Dot(axes=-1, normalize='l2')([embed_1, embed_3])

            #kld_term_1 = Lambda(lambda s : -kld_gauss(s))([embed_2, sigma_2, embed_1, sigma_1])
            #kld_term_2 = Lambda(kld_gauss)([embed_2, sigma_2, embed_1, sigma_1])

            #kld_term_2 = Lambda(lambda s : kld_gauss(s))([embed_3, sigma_3, embed_1, sigma_1])  

            vaeterms = Add()([vae_1, vae_2, vae_3])
            #kld_terms = Add()([kld_term_1, kld_term_2])

            model_pred = Model([input_1, input_2, input_3], [diff1, diff2, vaeterms])#, kld_terms])
            model_pred.name = 'pred_'+aspect
            models_pred[aspect] = model_pred

        P = OrderedDict()
        for aspect in self.aspects :
            P[aspect] = models_pred[aspect]([I[(aspect, 'O')], I[(aspect, 'S')], I[(aspect, 'D')]])
         
        D = OrderedDict()
        self.losses = {}

        for aspect in self.aspects :
            name_s = 'O_S'+aspect+'_score'
            name_v = 'V'+aspect
            name_w = 'K'+aspect
            
            output = P[aspect]

            D[name_s] = Lambda(lambda s : s[0] - s[1], name=name_s)([output[0], output[1]])
            D[name_v] = Activation('linear', name=name_v)(output[2])
            #D[name_w] = Activation('linear', name=name_w)(output[3])

            self.losses[name_s] = contrastive_loss
            self.losses[name_v] = lambda y_true, y_pred : 0.001 * K.mean(y_pred)
            #self.losses[name_w] = contrastive_loss

        self.model = Model(inputs=I.values(), outputs=D.values())
            
        self.model.compile(optimizer=Adam(lr=0.00005), loss=self.losses)
        
    def generate_y_batch(self, nb_sample) :
        y_batch = {}

        ones = []
        zeros = []
        for aspect in self.aspects :
            name_s = 'O_S'+aspect+'_score'
            name_d = 'O_D'+aspect+'_score'
            name_v = 'V' + aspect
            ones.append(name_s)
            zeros.append(name_d)
            zeros.append(name_v)
            zeros.append('K'+aspect)


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
