#==============================================================================#
#
# Jacob Sachs
# Computational Linguistics
# Finding Compounds
#
# Module to determine compounds within an English corpus.
#
#==============================================================================#

from __future__ import division
from   math     import log
from   sets     import Set


class CompoundFinder:
    """
    Constructor
    """
    def __init__(self):
    	self.occur = {}      # a dictionary of letter occurences
    	self.freq = {}       # a dictionary of letter frequencies
    	self.pair_occur = {} # a dictionary of letter occurences for bigrams
    	self.pair_freq = {}  # a dictionary of letter frequencies for bigrams

    	self.lexicon = Set()    # the list of words from our corpus
    	self.compounds = Set()  # a list of compound words determined by the algorithm

    	self.prefixes = Set([
    		'dis', 'in', 'im', 're', 'un']) # Taken from a list of most common prefixes
    	self.postfixes = Set([
    		'ed', 'ing', 'ly', 's', 'es']) # Most common postfixes

    """
    Function to load the lexicon from the corpus file
    """
    def get_lex(self, f):
    	lines = open(f).readlines()
    	for line in lines:
    		word = line.rstrip('\r\n')
    		self.lexicon.add(word)
    	return

    """
    Function to get unigram and bigram frequencies from a corpus
    """
    def get_freq(self):

        # Unigrams
        for word in self.lexicon:
            for letter in word:
                if letter.isalpha():
                    if letter not in self.occur:
                        self.occur[letter] = 1
                    else:
                        self.occur[letter] += 1
        for key, value in self.occur.iteritems():
            self.freq[key] = value/sum(self.occur.values())

        # Bigrams
        for word in self.lexicon:
            for i in range(len(word) - 1):
                key = word[i] + word[i+1]
                if key.isalpha():
                    if key not in self.pair_occur:
                        self.pair_occur[key] = 1
                    else:
                        self.pair_occur[key] += 1
        for key, value in self.pair_occur.iteritems():
            self.pair_freq[key] = value/sum(self.pair_occur.values())

        return

    """
    Helper function to print frequencies
    """
    def print_freq(self):
    	for key, value in self.freq.iteritems():
    		print key, value
    	for key, value in self.pair_freq.iteritems():
    		print key, value
    	return

    """
    Function to check the lexicon for words in a word

    This is currently the biggest cause of false positives,
    so while recall is decent, I'm getting a bunch of noise.
    """
    def check_lex(self, word):
        for i in range(1, len(word) - 1):
            if word[:i] in self.lexicon and word[i:] in self.lexicon:
                if word[:i] not in self.prefixes and word[i:] not in self.postfixes:
                    if len(word[:i]) > 1 and len(word[i:]) > 1:
                        self.compounds.add(word[:i] + ' ' + word[i:])
                        return

    """
    Function to calculate pointwise mutual information
    """
    def check_pmi(self, word):
        for i in range(1, len(word) - 1):
            a = word[i]
            b = word[i+1]

            if a+b in self.pair_freq and a in self.freq and b in self.freq:
                num = self.pair_freq[a+b]
                denom = self.freq[a] * self.freq[b]
                pmi = log(num/denom, 2)
                if pmi < -10: # Chosen by testing different values
                    if len(word[:i]) > 1 and len(word[:i]) > 1: # added to avoid false positives
                        self.compounds.add(word[:i] + ' ' + word[i:])
                        return

    """
    Revised algorithm to find compounds
    """
    def find_compounds(self):

        # Check for hyphens
        for word in self.lexicon:
            if '-' in word:
                self.compounds.add(word)

        # Check the lexicon
        for word in self.lexicon:
            if word in self.compounds:
                continue
            else:
                self.check_lex(word)
        
        
        # Check using PMI
        for word in self.lexicon:
            if word in self.compounds:
                continue
            else:
                self.check_pmi(word)
        """
        I thought this would be more useful, but it was
        a huge hit to recall, with a minor precision increase.

        # Clean out false positives using affixes
        for word in list(self.compounds):
            for pref in self.prefixes:
                if word.startswith(pref):
                    self.compounds.discard(word)
            for post in self.postfixes:
                if word.endswith(post):
                    self.compounds.discard(word)
        """

        return


if __name__ == '__main__':
    CF = CompoundFinder()
    corpus = 'browncorpus_simplewords.txt'
    CF.get_lex(corpus)
    CF.get_freq()
    CF.find_compounds()

    """
    Check against gold standard
    """
    standard = 'updated_gold_standard.txt'
    lines = open(standard).readlines()
    gs = Set()
    for line in lines:
        word = line.rstrip('\r\n')
        if ' ' in word or '-' in word:
            gs.add(word)

    correct = len(CF.compounds & gs)

    # Precision
    precision = "Precision: " + str(round(correct / len(CF.compounds), 4) * 100) + '%\n'

    # Recall
    recall = "Recall: " + str(round(correct / len(gs), 4) * 100) + '%\n'

    f = open('output.txt','w')
    f.write(precision)
    f.write(recall)
    for word in (CF.compounds & gs):
        f.write(word + '\n')
    f.close()

    print precision, recall


















