import pandas as pd

import pandas as pd
df = pd.read_csv('study_inclusion.csv', index_col=0)

dfj = pd.read_csv('test_cohen.csv', index_col=0)

fields = ['abstract', 'population', 'intervention', 'outcome']

texts = []
index = {}
for f in fields :
    text = list(df[f])
    orig_len =len(texts)
    texts += text
    index[f] = (orig_len, len(texts))
    

    
print index
from vectorizer import Vectorizer
from nltk import word_tokenize

texts = map(lambda s: ' '.join(word_tokenize(s)).lower(), texts)
texts = map(lambda s: ' '.join('qqq' if any(char.isdigit() for char in word) else word for word in s.split()), texts)

print "vectorizing"
import cPickle

vectorizer = Vectorizer()

vectorizer.fit(texts)
vectorizer.texts_to_sequences(texts)
vectorizer.index = index

cPickle.dump(vectorizer, open('allfields.p', 'wb'))