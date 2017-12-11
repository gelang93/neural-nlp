import numpy as np
import pandas as pd
import random

aspect_maps = sorted(['review/appearance', 'review/taste', 'review/aroma', 'review/palate'])
aspect_maps = {k:i for i, k in enumerate(aspect_maps)}

def posneg_matrix_generator(X, df, nb_sample, seed):
    random = np.random.RandomState(seed) if seed else np.random
    idxsidx = {}
    for aspect in aspect_maps :
        for i in [0, 1] :
            idxsidx[(aspect, i)] = df[df[aspect] == i].index

    bit_groups = list(idxsidx.keys())
    idxsidx[0] = df[df['review/overall'] < 3.0].index
    idxsidx[1] = df[df['review/overall'] > 3.5].index

    for aspect in aspect_maps :
        for i in [0, 1] :
            for j in [0, 1] :
                idxsidx[(aspect, i, j)] = list(set(idxsidx[(aspect, i)]) & set(idxsidx[j]))


    samples_per_group = nb_sample/2
        
    while True:
        X_orig = {'O'+str(v):[] for k, v in aspect_maps.items()}
        X_diff = {'D'+str(v):[] for k, v in aspect_maps.items()}
        X_v = {'V'+str(v) : [] for k, v in aspect_maps.items()}
        X_same = {'S'+str(v):[] for k, v in aspect_maps.items()}
        
        for aspect in aspect_maps :
            v = aspect_maps[aspect]
            for i in [0, 1] :
                orig = idxsidx[(aspect, i, i)]

                sample_size = min(len(orig), samples_per_group)
                orig = list(random.choice(orig, size=sample_size, replace=False))
                X_orig['O'+str(v)] += orig

                similar_1 = idxsidx[(aspect, i, 1-i)]
                similar_2 = idxsidx[(aspect, i, i)]
                different_1 = idxsidx[(aspect, 1-i, i)]
                different_2 = idxsidx[(aspect, 1-i, 1-i)]
                
                sample_size_1 = int(sample_size*0.75)
                sample_size_2 = sample_size - sample_size_1
                X_same['S'+str(v)] += list(random.choice(similar_1, size=sample_size_1, replace=True))
                X_same['S'+str(v)] += list(random.choice(similar_2, size=sample_size_2, replace=True))

                X_diff['D'+str(v)] += list(random.choice(different_1, size=sample_size_1, replace=True)) 
                X_diff['D'+str(v)] += list(random.choice(different_2, size=sample_size_2, replace=True))   
           
        X_batch = dict(X_orig.items() + X_diff.items() + X_same.items())
        
        for key in X_batch :
            X_batch[key] = X[X_batch[key]]
                    
        yield X_batch

from copy import deepcopy
def bg1(X, df, trainer, nb_sample=128, seed=1337):
    """O + D[0,1,2,3] + S[0,1,2,3]"""

    batch = posneg_matrix_generator(X, df, nb_sample, seed)
    while True:
        X_batch = next(batch)  
        y_batch = trainer.generate_y_batch(nb_sample)   
        
        yield X_batch, y_batch
