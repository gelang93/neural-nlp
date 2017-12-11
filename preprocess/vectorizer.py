import pickle

import numpy as np
import pandas as pd

from keras.preprocessing import sequence

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words
from nltk.tokenize import word_tokenize

from scipy.sparse import csr_matrix
from math import ceil

import spacy
nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
print nlp.pipeline

class Vectorizer:
    """Tiny class for fitting a vocabulary and vectorizing texts.

    Assumes texts have already been tokenized and that tokens are separated by
    whitespace.

    This class maintains state so that it can be pickled and used to train keras
    models.

    """
    def __init__(self, num_words=None, min_df=None):
        self.embeddings = None
        self.word_dim = 200
        self.num_words = num_words
        self.min_df = min_df

    def __len__(self):
        """Return the length of X"""

        return len(self.X)

    def __getitem__(self, given):
        """Return a slice of X"""

        return self.X[given]
    
    def tokenizer(self, text, which=False) :
        if which :
            text = [t.text for t in nlp(unicode(text, 'utf-8').lower())]
        else :
            text = [t.text for t in nlp(unicode(text).lower())]
        #text = word_tokenize(text.lower())
        text = ['qqq' if any(char.isdigit() for char in word) else word for word in text]
        return text
    
    def convert_to_sequence(self, texts) :
        texts_tokenized = map(lambda s : self.tokenizer(s, which=True), texts)
        texts_tokenized = map(lambda s : ['unk' if word not in self.word2idx else word for word in s], texts_tokenized)
        sequences = map(lambda s : [self.word2idx[word] for word in s], texts_tokenized)
        return sequences, texts_tokenized

    def fit(self, texts):
        """Fit the texts with a keras tokenizer
        
        Parameters
        ----------
        texts : list of strings to fit and vectorize
        word_list : list of words to include in the text
        
        """
        cvec = CountVectorizer(tokenizer=self.tokenizer, min_df=self.min_df)#, max_features=self.num_words)
        cvec.fit(texts)
                
        self.word2idx = cvec.vocabulary_
        for word in self.word2idx :
            self.word2idx[word] += 2
            
        self.word2idx['[0]'] = 0
        self.word2idx['unk'] = 1
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
   
        self.fit_texts = texts

        self.vocab_size = len(self.word2idx)


    def texts_to_sequences(self, texts, do_pad=True, maxlen=None, maxlen_ratio=0.95):
        """Vectorize texts as sequences of indices
        
        Parameters
        ----------
        texts : pd.Series of strings to vectorize into sequences of indices
        do_pad : pad the sequences to `self.maxlen` if true
        maxlen : maximum length for texts
        maxlen_ratio : compute maxlen M dynamically as follows: M is the minimum
        number such that `maxlen_ratio` percent of texts have length greater
        than or equal to M.

        First replace OOV words with unk token.

        """
        self.X, texts_tok = self.convert_to_sequence(texts)

        if do_pad:
            if not maxlen:
                lengths = [len(text) for text in texts_tok]
                self.maxlen = int(ceil(np.percentile(lengths, maxlen_ratio*100)))
            else:
                self.maxlen = maxlen

            self.X = sequence.pad_sequences(self.X, maxlen=self.maxlen)

        return self.X

    def extract_embeddings(self, model):
        """Pull out pretrained word vectors for every word in vocabulary
        
        Parameters
        ----------
        model : gensim word2vec model

        If word is not in `model`, then randomly initialize it.
        
        """
        self.word_dim, self.vocab_size = model.vector_size, len(self.word2idx)
        self.embeddings = np.zeros([self.vocab_size, self.word_dim])
        in_pre = 0
        for i, word in sorted(self.idx2word.items()):
            if word in model :
                self.embeddings[i] = model[word] 
                in_pre += 1
            else :
                self.embeddings[i] = np.random.randn(self.word_dim)
                
        print "Found " + str(in_pre) + " words in pubmed out of " + str(len(self.idx2word))
        return self.embeddings

    def test(self, doc_idx):
        """Recover text from vectorized representation of the `doc_idx`th text

        Parameters
        ----------
        doc_idx : document index to recover text from

        This function is just for sanity checking that sequence vectorization
        works.

        """
        print self.X[doc_idx]
        print
        print ' '.join(self.idx2word[idx] for idx in self.X[doc_idx] if idx)
