import cPickle

import sys
sys.path.insert(0, '../../preprocess')
import vectorizer

cdsr_data = cPickle.load(open('../../beer_data/beer_vec_ds_df10.p', 'rb'))

idx2word = cdsr_data.idx2word
X = cdsr_data.X

X = map(lambda s : [idx2word[t] for t in s if t > 0], X)

print X[0]

from gensim.models.doc2vec import LabeledSentence, Doc2Vec
sent_it = (LabeledSentence(s, [str(i)]) for i, s in enumerate(X))
model = Doc2Vec(size=800, window=10, min_count=5, workers=4, alpha = 0.025, min_alpha=0.025)

model.build_vocab(sent_it)

sent_it = (LabeledSentence(s, [str(i)]) for i, s in enumerate(X))
print model.corpus_count
for epoch in range(10) :
    model.train(sent_it, total_examples=model.corpus_count, epochs=1)
    print "Epoch ", epoch
    model.alpha -= 0.002
    model.min_alpha = model.alpha
    
model.save('../store/doc2vec_model_800_beer_df10.mod')

         