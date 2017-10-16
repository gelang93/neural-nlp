from collections import OrderedDict

import keras.backend as K
from keras.layers import Input, Embedding, Dropout, Dense, LSTM, merge, Lambda, Concatenate
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, merge
from keras.layers import Activation, Lambda, ActivityRegularization
from keras.layers.merge import Dot
from keras.models import Model
from keras.regularizers import l2

from trainer import Trainer
from support import cnn_embed, lstm_embed

import numpy as np

class SharedCNNSiameseTrainer(Trainer):
    def build_model(self, nb_filter=300, filter_lens=range(1,4), reg=0.0001):
        aspect = self.C['aspect']
        inputs = ['same_abstract']
        scores = []
        # for mod in ['valid', 'corrupt'] :
        #     inputs += [mod + '_abstract']
        #     scores += [mod + '_abstract']
            
        for mod in self.modifier :
            inputs += [mod + '_' + aspect]
            scores += [mod + '_' + aspect]
        
        for mod in ['same'] :
            for other_aspect in self.C['aspect_comp'] :
                inputs += [mod + '_' + other_aspect]
                scores += [mod + '_' + other_aspect]
        
        maxlen = self.vec.maxlen
        word_dim, vocab_size = self.vec.word_dim, self.vec.vocab_size

        input = Input(shape=[maxlen], dtype='int32')
        lookup = Embedding(output_dim=word_dim, 
                           input_dim=vocab_size, 
                           name='embedding', 
                           weights=[self.vec.embeddings])(input)
        cnn_network = cnn_embed(lookup, filter_lens, nb_filter, maxlen, word_dim, reg, name='pool')
        lstm_network = lstm_embed(lookup, 128, reg)
        model = Model(input, cnn_network)
        model.name = 'pool'
        
        I = OrderedDict() # inputs
        for s in inputs:
            I[s] = Input(shape=[maxlen], dtype='int32', name=s)
        
        C = OrderedDict()
        for s in inputs:
            C[s] = model(I[s])

        D = OrderedDict() # dots
        for s in scores:
            D[s] = Dot(axes=1, name=s+'_score')([C['same_abstract'], C[s]])

        self.model = Model(inputs=I.values(), outputs=D.values())
        self.losses = {}
        for s in scores :
            self.losses[s + '_score'] = 'hinge'
            
        self.loss_weights = {}
        for loss in self.losses :
            self.loss_weights[loss] = 1.0# K.variable(1.0)
            
        for mod in self.modifier :
            loss = mod + '_' + aspect + '_score'
            self.loss_weights[loss] = 5.0
            
        
        self.zero_what = []
        # for aspect in self.C['aspect_comp'] :
        #     self.zero_what.append('same_' + aspect + '_score')
            
        self.zero_after = 10
        self.model.compile(optimizer='adam', loss=self.losses, loss_weights=self.loss_weights)
        
    def generate_y_batch(self, nb_sample) :
        aspect = self.C['aspect']
        y_batch = {}
        
        y_batch['same_' + aspect + '_score'] = np.ones(nb_sample)
        y_batch['valid_' + aspect + '_score'] = np.ones(nb_sample)
        y_batch['corrupt_' + aspect + '_score'] = np.full(shape=nb_sample, fill_value=-1)
        
        y_batch['corrupt_abstract_score'] = np.full(shape=nb_sample, fill_value=-1)
        y_batch['valid_abstract_score'] = np.ones(nb_sample)
        
        y_batch['valid_' + aspect + '_aspect_score'] = np.ones(nb_sample)
        y_batch['corrupt_' + aspect + '_aspect_score'] = np.full(shape=nb_sample, fill_value=-1)

        for a in self.C['aspect_comp'] :
            for mod in self.modifier :
                y_batch[mod + '_' + a + '_score'] = np.full(shape=nb_sample, fill_value=-1)
            
        return y_batch
    
class SingleCNNModel(Trainer) :
    def build_model(self, nb_filter=300, filter_lens=range(1,4), reg=0.0001):
        self.aspects = ['population', 'intervention', 'outcome']

        inputs = {}
        for aspect in self.aspects :
            for mod in self.modifier :
                inputs[(aspect, mod)] = mod + '_' + aspect
                
        inputs[('abstract', 'same')] = 'same_abstract'
        
        maxlen = self.vec.maxlen
        word_dim, vocab_size = self.vec.word_dim, self.vec.vocab_size

        input = Input(shape=[maxlen], dtype='int32')
        lookup = Embedding(output_dim=word_dim, 
                           input_dim=vocab_size, 
                           name='embedding', 
                           weights=[self.vec.embeddings])(input)
        
        models = OrderedDict()
        cnn_networks = []
        for aspect in self.aspects :
            cnn_network = cnn_embed(lookup, filter_lens, nb_filter, maxlen, word_dim, reg, name='pool_'+aspect)
            model = Model(input, cnn_network)
            model.name = 'pool_' + aspect
            models[aspect] = model
            cnn_networks.append(cnn_network)
            
        concat_layer = Concatenate()(cnn_networks)
        self.evaluation_model = Model(input, concat_layer)
        self.evaluation_model.name = 'pool'   
                    
        I = OrderedDict()
        for input in inputs :
            I[input] = Input(shape=[maxlen], dtype='int32', name=inputs[input])
            
        evaluation_layer = self.evaluation_model(I[('abstract', 'same')])
            
        C = OrderedDict()
        for aspect in self.aspects :
            for input in inputs :
                if input[0] in ['abstract', aspect] :
                    C[(input, aspect)] = models[aspect](I[input])     
                    
        D = OrderedDict()
        self.losses = {}
        for aspect in self.aspects :
            for mod in self.modifier :
                abstract_embedding = C[(('abstract', 'same'), aspect)]
                aspect_summ_embedding = C[((aspect, mod), aspect)]
                name = 'same_abstract_' + mod + '_' + aspect + '_score'
                D[(('abstract', 'same'), (aspect, mod))] = Dot(axes=1, name=name)([abstract_embedding, aspect_summ_embedding])
                self.losses[name] = 'hinge'
                
        D[('pool')] = Dot(axes=1, name='pool_score')([evaluation_layer, evaluation_layer])
                
        from itertools import combinations
        aspect_comb = list(combinations(self.aspects, 2))
        self.loss_weights = {}
        for comb in aspect_comb :
            abstract_emb_1 = C[(('abstract', 'same'), comb[0])]
            abstract_emb_2 = C[(('abstract', 'same'), comb[1])]
            name = 'same_' + comb[0] + '_same_' + comb[1] + '_score'
            D[((comb[0], 'same'),(comb[1], 'same'))] = Dot(axes=1, name=name)([abstract_emb_1, abstract_emb_2])
            self.losses[name] = 'hinge'
            self.loss_weights[name] = 2.0
            
        self.loss_weights['pool_score'] = 0.0
        
        self.losses['pool_score'] = 'hinge'
        self.model = Model(inputs=I.values(), outputs=D.values())
            
        self.model.compile(optimizer='adam', loss=self.losses, loss_weights=self.loss_weights)
        
    def generate_y_batch(self, nb_sample) :
        ones = []
        neg_ones = []
        for aspect in self.aspects :
            for mod in ['same', 'valid'] :
                name = 'same_abstract_' + mod + '_' + aspect + '_score'
                ones.append(name)
            mod = 'corrupt'
            name = 'same_abstract_' + mod + '_' + aspect + '_score'
            neg_ones.append(name)
            
        from itertools import combinations
        aspect_comb = list(combinations(self.aspects, 2))
        
        for comb in aspect_comb :
            name = 'same_' + comb[0] + '_same_' + comb[1] + '_score'
            neg_ones.append(name)
            
        ones.append('pool_score')
            
        y_batch = {}
        for loss in ones :
            y_batch[loss] = np.ones(nb_sample)
            
        for loss in neg_ones :
            y_batch[loss] = np.full(shape=nb_sample, fill_value=-1)
            
        return y_batch
#############################################################################################################################################################################################################################################################

###############################################################################################################################
class SingleCNNModel2(Trainer) :
    def build_model(self, nb_filter=300, filter_lens=range(1,4), reg=0.0001):
        self.aspects = ['population', 'intervention', 'outcome']

        inputs = {}
        for aspect in self.aspects :
            for mod in ['same'] :
                inputs[(aspect, mod)] = mod + '_' + aspect
                
        for mod in self.modifier :
            inputs[('abstract', mod)] = mod + '_abstract'
        
        maxlen = self.vec.maxlen
        word_dim, vocab_size = self.vec.word_dim, self.vec.vocab_size

        input = Input(shape=[maxlen], dtype='int32')
        lookup = Embedding(output_dim=word_dim, 
                           input_dim=vocab_size, 
                           name='embedding', 
                           weights=[self.vec.embeddings])(input)
        
        models = OrderedDict()
        cnn_networks = []
        for aspect in self.aspects :
            cnn_network = cnn_embed(lookup, filter_lens, nb_filter, maxlen, word_dim, reg, name='pool_'+aspect)
            model = Model(input, cnn_network)
            model.name = 'pool_' + aspect
            models[aspect] = model
            cnn_networks.append(cnn_network)
            
        concat_layer = Concatenate()(cnn_networks)
        self.evaluation_model = Model(input, concat_layer)
        self.evaluation_model.name = 'pool'   
                    
        I = OrderedDict()
        for input in inputs :
            I[input] = Input(shape=[maxlen], dtype='int32', name=inputs[input])
            
        evaluation_layer = self.evaluation_model(I[('abstract', 'same')])
            
        C = OrderedDict()
        for aspect in self.aspects :
            for input in inputs :
                if input[0] in ['abstract', aspect] :
                    C[(input, aspect)] = models[aspect](I[input])     
                    
        D = OrderedDict()
        self.losses = {}
        for aspect in self.aspects :
            embed_1 = C[(('abstract', 'same'), aspect)]
            embed_2 = C[((aspect, 'same'), aspect)]
            name = 'same_abstract_same_' + aspect + '_score'
            D[(('abstract', 'same'), (aspect, 'same'))] = Dot(axes=1, name=name)([embed_1, embed_2])
            self.losses[name] = 'hinge'
            for mod in ['valid', 'corrupt'] :
                embed_2 = C[(('abstract', mod), aspect)]
                name = 'same_abstract_' + mod + '_abstract_' + aspect + '_score'
                D[(('abstract', 'same'), ('abstract', mod), aspect)] = Dot(axes=1, name=name)([embed_1, embed_2])
                self.losses[name] = 'hinge'
                
        D[('pool')] = Dot(axes=1, name='pool_score')([evaluation_layer, evaluation_layer])
                
        from itertools import combinations
        aspect_comb = list(combinations(self.aspects, 2))
        for comb in aspect_comb :
            abstract_emb_1 = C[(('abstract', 'same'), comb[0])]
            abstract_emb_2 = C[(('abstract', 'same'), comb[1])]
            name = 'same_' + comb[0] + '_same_' + comb[1] + '_score'
            D[((comb[0], 'same'),(comb[1], 'same'))] = Dot(axes=1, name=name)([abstract_emb_1, abstract_emb_2])
            self.losses[name] = 'hinge'
        
        self.losses['pool_score'] = 'hinge'
        self.model = Model(inputs=I.values(), outputs=D.values())
            
        self.model.compile(optimizer='adam', loss=self.losses, loss_weights={'pool_score': 0.0})
        
    def generate_y_batch(self, nb_sample) :
        ones = []
        neg_ones = []
        for aspect in self.aspects :
            ones.append('same_abstract_same_' + aspect + '_score')
            ones.append('same_abstract_valid_abstract_' + aspect + '_score')
            neg_ones.append('same_abstract_corrupt_abstract_' + aspect + '_score')
            
        from itertools import combinations
        aspect_comb = list(combinations(self.aspects, 2))
        
        for comb in aspect_comb :
            name = 'same_' + comb[0] + '_same_' + comb[1] + '_score'
            neg_ones.append(name)
            
        ones.append('pool_score')
            
        y_batch = {}
        for loss in ones :
            y_batch[loss] = np.ones(nb_sample)
            
        for loss in neg_ones :
            y_batch[loss] = np.full(shape=nb_sample, fill_value=-1)
            
        return y_batch