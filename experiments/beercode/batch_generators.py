import numpy as np
import pandas as pd
import random

aspect_maps = sorted(['review/appearance', 'review/taste', 'review/aroma', 'review/palate'])
aspect_maps = {k:i for i, k in enumerate(aspect_maps)}

def search_comp(code, aspect, bit_groups) :
    code_2 = list(np.random.choice([0, 1], size=(4,)))
    code_2[aspect_maps[aspect]] = code[aspect_maps[aspect]]
    return code_2

def flip_aspect(code, aspect, bit_groups) :
    code_2 = list(code)
    #code_2 = search_comp(code, aspect, bit_groups)
    code_2[aspect_maps[aspect]] = 1 - code[aspect_maps[aspect]]
    return tuple(code_2)

def keep_aspect(code, aspect, bit_groups) :
    return code

def flip_aspect_comp(code, aspect, bit_groups) :
    code_2 = search_comp(code, aspect, bit_groups)
    code_3 = list(code_2)
    #code_3 = [1-code[aspect_maps[aspect]] for x in code_3]
    code_3[aspect_maps[aspect]] = 1 - code_3[aspect_maps[aspect]]
    return tuple(code_2), tuple(code_3) 

def flip_aspect_all(code, aspect, bit_groups) :
    code_2 = [1-x for x in code]
    code_2[aspect_maps[aspect]] = code[aspect_maps[aspect]]
    code_3 = list(code)
    code_3[aspect_maps[aspect]] = 1 - code[aspect_maps[aspect]]
    return tuple(code_2), tuple(code_3)



def cdno_matrix_generator(X, df, nb_sample, seed):
    random = np.random.RandomState(seed) if seed else np.random
    review_index = df.index
    
    bit_groups = list(df.bits.unique())

    group2study_idxs = {x:df[df.bits == x].index for x in bit_groups}
    samples_per_group = nb_sample/len(bit_groups)
        
    while True:
        X_orig = {'O':[]}
        X_diff = {'D'+str(v):[] for k, v in aspect_maps.items()}
        X_same = {'S'+str(v):[] for k, v in aspect_maps.items()}
        y = {str(v)+'_pred' : [] for k, v in aspect_maps.items()}
        
        for g in bit_groups :
            sample_size = min(len(group2study_idxs[g]), samples_per_group)
            X_orig['O'] += list(random.choice(group2study_idxs[g], size=sample_size, replace=False))
            for aspect in aspect_maps :
                v = aspect_maps[aspect]
                y[str(v)+'_pred'] += [[1-g[v],g[v]] for _ in range(sample_size)]
                #same_aspect_code, diff_aspect_code = flip_aspect_comp(g, aspect, bit_groups)
                diff_aspect_code = flip_aspect(g, aspect, bit_groups)
                X_diff['D'+str(v)] += list(random.choice(group2study_idxs[diff_aspect_code], size=sample_size, replace=True))
                
                same_aspect_code = keep_aspect(g, aspect, bit_groups)
                X_same['S'+str(v)] += list(random.choice(group2study_idxs[same_aspect_code], size=sample_size, replace=True))
                

        X_batch = dict(X_orig.items() + X_diff.items() + X_same.items())
        for key in X_batch :
            X_batch[key] = X[X_batch[key]]
            
        for key in y :
            y[key] = np.array(y[key])
            
        yield X_batch, y

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
        X_same = {'S'+str(v):[] for k, v in aspect_maps.items()}
        y = {str(v)+'_pred' : [] for k, v in aspect_maps.items()}
        
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
            
        yield X_batch, y

def bg1(X, df, trainer, nb_sample=128, seed=1337):
    """O + D[0,1,2,3] + S[0,1,2,3]"""

    batch = posneg_matrix_generator(X, df, nb_sample, seed)
    while True:
        X_batch, y_batch = next(batch)  
        y_batch = trainer.generate_y_batch(nb_sample)   
        # for key in y_batch :
        #     y_batch[key] = y_batch[key][:len(X_batch['O0'])]
        yield X_batch, y_batch
