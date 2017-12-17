
# coding: utf-8

# In[1]:


import cPickle
import sys
sys.path.insert(0, '../../preprocess')
import vectorizer

vec = cPickle.load(open('../../preprocess/allfields_with_embedding_19995.p', 'rb'))
cohen_vec = cPickle.load(open('../../preprocess/cohendata_dedup_19995.p'))


# In[2]:


index = vec.index['abstract']
vec.X = vec.X[index[0]:index[1]]


# In[3]:


import numpy as np
train_X = vec.X
X_tf = np.zeros((train_X.shape[0], vec.vocab_size))
for i in range(len(train_X)) :
    X_tf[i, train_X[i, :]] = 1.

X_tf = X_tf[:, 2:]
train_Xtf = X_tf

X = cohen_vec.X
cohen_X_tf = np.zeros((X.shape[0], vec.vocab_size))
for i in range(len(X)) :
    cohen_X_tf[i, X[i, :]] = 1.

cohen_X_tf = cohen_X_tf[:, 2:]

import pandas as pd
df = pd.read_csv('../data/files/test_cohen_dedup.csv')

nb_studies = len(df)
H = np.zeros((nb_studies, nb_studies))

cdnos = list(set(df.cdno))
for i in range(nb_studies) :
    H[i, df[df['cdno'] == df['cdno'][i]].index] = 1
    
H[np.arange(nb_studies), np.arange(nb_studies)] = 0

# In[4]:


from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras import optimizers

learning_rate = 5e-5
batch_size = 128
vocab_size = vec.vocab_size - 2
intermediate_dim = 500
latent_dim = 200
epochs =25 
epsilon_std = 1.0
activation = 'tanh'

for n_top in range(75, 300, 5) :
    latent_dim = n_top
    x = Input(shape=(vocab_size,), name='x')
    h = Dense(intermediate_dim, activation=activation, name='h')(x)
    mu = Dense(latent_dim, name='mu')(h)
    log_sigma2 = Dense(latent_dim, name='l')(h)
    encoder = Model(x, mu)

    # reparameterized sampler for normal distributions
    def sample_norm(args):
        '''reparameterized sampling from normal distribution'''
        mu, log_var = args
        epsilon = K.random_normal(shape=(K.shape(mu)[0], latent_dim,), mean=0.)
        return mu + K.exp(0.5 * log_var) * epsilon

    # decoder / generative network
    z = Lambda(sample_norm, output_shape=(latent_dim,), name='z')([mu, log_sigma2])
    e = Dense(vocab_size, name='e')(z)

    def log_softmax(x, axis=None):
        x0 = x - K.max(x, axis=axis, keepdims=True)
        log_sum_exp_x0 = K.log(K.sum(K.exp(x0), axis=axis, keepdims=True))
        return x0 - log_sum_exp_x0

    def kl_loss(x, e): 
        return (- 0.5 * K.sum(1 + log_sigma2 - K.square(mu) - K.exp(log_sigma2), axis=-1))


    def cross_ent_loss(x, e): 
        return - K.sum(x * log_softmax(e, axis=-1), axis=-1) 
        

    def vae_loss(x, e):
        xent_loss = cross_ent_loss(x, e)
        kld = kl_loss(x, e)
        return xent_loss + kld


    opt = optimizers.adam(lr=learning_rate)
    vae = Model(x, e)
    vae.compile(optimizer=opt, 
                loss=vae_loss)


    from keras import callbacks
    patience = 3
    earlyStopping = callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min')

    vae.fit(train_Xtf,  
            train_Xtf, 
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0, 
            callbacks=[earlyStopping], 
            validation_split=0.1)

    embedds = encoder.predict(cohen_X_tf)

    from sklearn.preprocessing import normalize
    embedds_n = normalize(embedds, 'l2')
    scores = np.dot(embedds_n, embedds_n.T)
    scores[np.arange(nb_studies), np.arange(nb_studies)] = -1000

    from sklearn.metrics import roc_auc_score
    aucs = [0] * nb_studies
    for i in range(nb_studies) :
        aucs[i] = roc_auc_score(H[i], scores[i])
    print n_top, np.mean(aucs)



