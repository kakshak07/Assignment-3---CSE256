#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from collections import defaultdict
import collections
from math import log
import sys

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        return pow(2.0, self.entropy(corpus))

    def entropy(self, corpus):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i])
        p += self.cond_logprob('END_OF_SENTENCE', sentence)
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous): pass
    # required, the list of words the language model suports (including EOS)
    def vocab(self): pass

class Unigram(LangModel):
    def __init__(self, backoff = 0.000001):
        self.model = dict()
        self.lbackoff = log(backoff, 2)

    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot

    def cond_logprob(self, word, previous):
        if word in self.model:
            return self.model[word]
        else:
            return self.lbackoff

    def vocab(self):
        return self.model.keys()


class Trigram(LangModel):
    def __init__(self, backoff = 0.000001, delta=0.001):
        self.model = dict()
        self.tri=dict()
        self.bi=dict()
        self.uni=dict()
        self.delta=delta
        self.lbackoff = log(backoff, 2)

    def inc_tri(self, w):
        if w in self.tri:
            self.tri[w] += 1.0
        else:
            self.tri[w] = 1.0
    def inc_bi(self, w):
        if w in self.bi:
            self.bi[w] += 1.0
        else:
            self.bi[w] = 1.0
    def inc_uni(self, w):
        if w in self.uni:
            self.uni[w] += 1.0
        else:
            self.uni[w] = 1.0

    def fit_sentence(self, sentence):
        sentence=["START_OF_SENTENCE","START_OF_SENTENCE"]+sentence+["END_OF_SENTENCE"]
        for i in range(len(sentence)):
            self.inc_uni(sentence[i])
        for i in range(len(sentence)-2):
            self.inc_tri((sentence[i],sentence[i+1],sentence[i+2]))
        for i in range(len(sentence)-1):
            self.inc_bi((sentence[i],sentence[i+1]))


    def norm(self):
        for i in self.tri:
            self.model[i]=log(self.tri[i]+self.delta, 2) - log(self.bi[(i[0],i[1])]+self.delta*len(self.vocab()), 2)
   

    def cond_logprob(self, word, previous):
        if not previous:
            logprobtri=("START_OF_SENTENCE","START_OF_SENTENCE",word)
        elif(len(previous)==1):
            logprobtri=("START_OF_SENTENCE",previous[0],word)
        else:
            logprobtri=(previous[-2],previous[-1],word)
        if(logprobtri in self.model):
            return(self.model[logprobtri])
        else:
            if(logprobtri[:2] in self.bi):
                return(log(self.delta, 2) - log(self.bi[logprobtri[:2]]+self.delta*len(self.vocab()), 2))
            else:
                return(- log(len(self.vocab()), 2))

    def vocab(self):
        return self.uni.keys()

