import pandas as pd

import pandas as pd
df = pd.read_csv('study_inclusion.csv', index_col=0)

fields = ['abstract', 'population', 'intervention', 'outcome']

texts = []
index = {}
for f in fields :
    text = list(df[f])
    texts += text
    index[f] = len(texts)
    
print index
from vectorizer import Vectorizer
from nltk import word_tokenize

texts = map(lambda s: ' '.join(word_tokenize(s)).lower(), texts) # basic tokenization
texts = map(lambda s: ' '.join('qqq' if any(char.isdigit() for char in word) else word for word in s.split()), texts)

print "vectorizing"
import cPickle

vectorizer = Vectorizer()

vectorizer.fit(texts)
vectorizer.texts_to_sequences(texts)

cPickle.dump([vectorizer, index], open('allfields.p', 'wb'))