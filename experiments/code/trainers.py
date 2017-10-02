from collections import OrderedDict

import keras.backend as K
from keras.layers import Input, Embedding, Dropout, Dense, LSTM, merge, Lambda
from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten, merge
from keras.layers import Activation, Lambda, ActivityRegularization
from keras.layers.merge import Dot
from keras.models import Model
from keras.regularizers import l2

from trainer import Trainer
from support import cnn_embed, average, norm2d


class CNNSiameseTrainer(Trainer):
    def build_model(self, nb_filter=300, filter_lens=range(5), dropout_emb=0.2, reg=0.001, update_embeddings=True, project_summary=False):
        A, S, O = self.A, self.S, self.O
        
        I = OrderedDict() # inputs
        for s in A+S:
            maxlen = self.C[s[1]].maxlen
            I[s[0]] = Input(shape=[maxlen], dtype='int32', name=s[0])

        W = OrderedDict() # words
        for s in A+S:
            word_dim, vocab_size = self.C[s[1]].word_dim, self.C[s[1]].vocab_size
            lookup = Embedding(output_dim=word_dim, input_dim=vocab_size, name='embedding_' + s[0])
            W[s[0]] = lookup(I[s[0]])

        C = OrderedDict()
        for s in A+S:
            maxlen, word_dim = self.C[s[1]].maxlen, self.C[s[1]].word_dim
            C[s[0]] = cnn_embed(filter_lens, nb_filter, maxlen, word_dim, reg, name='pool_' + s[0])(W[s[0]])

        D = OrderedDict() # dots
        for s in S:
            D[s[0]] = Dot(axes=1, name=s[0]+'_score')([C['same_abstract'], C[s[0]]])

        self.model = Model(inputs=I.values(), outputs=D.values())
        aspect = self.C['aspect']
        losses = {'same_'+aspect+'_score': 'hinge',
                  'valid_'+aspect+'_score': 'hinge',
                  'corrupt_'+aspect+'_score': 'hinge'
        }
        self.model.compile(optimizer='adam', loss=losses)


class SharedCNNSiameseTrainer(Trainer):
    """Two-input model which embeds abstract and target
    
    Either push them apart or pull them together depending on label. The two
    models share weights as the inputs come from the same distribution.

    """
    def build_model(self, nb_filter=300, filter_lens=range(1,3), dropout_emb=0.2, reg=0.001):
        A, S, O = self.A, self.S, self.O
        maxlen = self.vec.maxlen
        word_dim, vocab_size = self.vec.word_dim, self.vec.vocab_size

        input = Input(shape=[maxlen], dtype='int32')
        lookup = Embedding(output_dim=word_dim, input_dim=vocab_size, name='embedding')(input)
        cnn_network = cnn_embed(lookup, filter_lens, nb_filter, maxlen, word_dim, reg, name='pool')
        model = Model(input, cnn_network)
        model.name = 'pool'
        
        I = OrderedDict() # inputs
        for s in A+S:
            I[s[0]] = Input(shape=[maxlen], dtype='int32', name=s[0])
        
        C = OrderedDict()
        for s in A+S:
            C[s[0]] = model(I[s[0]])

        D = OrderedDict() # dots
        for s in S:
            D[s[0]] = Dot(axes=1, name=s[0]+'_score')([C['same_abstract'], C[s[0]]])

        self.model = Model(inputs=I.values(), outputs=D.values())
        aspect = self.C['aspect']
        losses = {'same_'+aspect+'_score': 'hinge',
                  'valid_'+aspect+'_score': 'hinge',
                  'corrupt_'+aspect+'_score': 'hinge'
        }
        
        self.model.compile(optimizer='adam', loss=losses)

class AdversarialTrainer(Trainer):
    """Six-input model of abstract and aspect summaries

    The model takes in an abstract and an aspect. We are given a "valid" aspect
    summary (from the same review) and a "corrupt" aspect summary (from a
    different review). We are also given the a "same" aspect summary (for the
    given abstract) along with "same" summaries for the other abstracts (for the
    given abstract).

    We run a conv over the abstract and summaries for the given aspect. We also
    compute the norm of the embeddings for each summary so we can either enforce
    them to be big (for the words in the summaries for the aspect) or small (for
    words in other aspect summaries).
    
    """
    def build_model(self):
        maxlen = self.C['maxlen']
        aspect = self.C['aspect']
        A, S, O = self.A, self.S, self.O

        I = OrderedDict() # inputs
        for s in A+S+O:
            maxlen = self.C[s[1]].maxlen
            I[s[0]] = Input(shape=[maxlen], dtype='int32', name=s[0])

        W = OrderedDict() # words
        for s in A+S+O:
            word_dim, vocab_size = self.C[s[1]].word_dim, self.C[s[1]].vocab_size
            lookup = Embedding(output_dim=word_dim, input_dim=vocab_size, name='embedding_' + s[0])
            W[s[0]] = lookup(I[s[0]])

        C, P = OrderedDict(), OrderedDict() # conv and pool
        convolve, pool = Conv1D(filters=100, kernel_size=1), GlobalMaxPooling1D(name='pool')
        for s in A+S:
            C[s[0]] = convolve(W[s[0]])
            P[s[0]] = pool(C[s[0]])

        D = OrderedDict() # dots
        for s in S:
            D[s[0]] = Dot(axes=1, name=s[0]+'_score')([P['same_abstract'], P[s[0]]])

        N = OrderedDict() # norms
        for s in S+O:
            N[s[0]] = Lambda(norm2d, output_shape=[1], name=s[0]+'_norm')(W[s[0]])

        for s in S:
            N[s[0]] = Lambda(lambda x: -x, name='neg_'+s[0]+'_norm')(N[s[0]])

        self.model = Model(inputs=I.values(), outputs=D.values())#+N.values())
        identity = lambda y_true, y_pred: K.mean(y_pred)
        aspect = self.C['aspect']
        losses = {'same_'+aspect+'_score': 'hinge',
                  'valid_'+aspect+'_score': 'hinge',
                  'corrupt_'+aspect+'_score': 'hinge',
                  # 'neg_same_'+aspect+'_norm': identity,
                  # 'neg_valid_'+aspect+'_norm': identity,
                  # 'neg_corrupt_'+aspect+'_norm': identity,
                  # 'same_intervention_norm': identity,
                  # 'same_outcome_norm': identity,
        }
        self.model.compile(optimizer='adam', loss=losses)
