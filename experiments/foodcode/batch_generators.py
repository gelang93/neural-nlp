import numpy as np
import pandas as pd
import random

from itertools import product

aspects = {'bit' : 0, 'domain' : 1}

def cdno_matrix_generator(X, df, nb_sample, seed):
    random = np.random.RandomState(seed) if seed else np.random    
    aspect_vals = {v:df[k].unique() for k, v in aspects.items()}

    groups = list(product(*[aspect_vals[v] for k, v in aspects.items()]))
    
    group2idxs = {x:df[df.apply(lambda y : all(y[k] == x[aspects[k]] for k in aspects), axis=1)].index for x in groups}

    samples_per_group = nb_sample/len(groups)
        
    while True:
        X_orig = {'O':[]}
        X_diff = {'D'+k:[] for k in aspects}
        X_same = {'S'+k:[] for k in aspects}
        
        for g in groups :
            X_orig['O'] += list(random.choice(group2idxs[g], size=samples_per_group, replace=False))
            for aspect in aspects :
                v = aspects[aspect]
                other_v = 1 - aspects[aspect]
                curr = g[v]
                diff = random.choice(list(set(aspect_vals[v]) - set([curr])))
                other_aspect_val = random.choice(aspect_vals[other_v])
                gs = [0,0]
                gd = [0,0]
                gs[v], gd[v] = curr, diff
                gs[other_v], gd[other_v] = other_aspect_val, other_aspect_val
                X_diff['D'+aspect] += list(random.choice(group2idxs[tuple(gd)], size=samples_per_group, replace=False))
                X_same['S'+aspect] += list(random.choice(group2idxs[tuple(gs)], size=samples_per_group, replace=False))

        X_batch = dict(X_orig.items() + X_diff.items() + X_same.items())
        for key in X_batch :
            X_batch[key] = X[X_batch[key]]
            
        yield X_batch

def bg1(X, df, trainer, nb_sample=128, seed=1337):
    """O + D[0,1,2,3] + S[0,1,2,3]"""

    batch = cdno_matrix_generator(X, df, nb_sample, seed)
    while True:
        X_batch = next(batch)  
        y_batch = trainer.generate_y_batch(nb_sample)   
        for key in y_batch :
            y_batch[key] = y_batch[key][:len(X_batch['O'])]
        yield X_batch, y_batch
