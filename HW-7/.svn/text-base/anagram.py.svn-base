#------------------------------------------------------------------------------#
#
# Jacob Sachs
# Anagrams
#
#
#------------------------------------------------------------------------------#

from sets import Set
import operator

class Anagram:
    def __init__(self, f):
        lines = open(f,'r').readlines()
        self.words = []
        for line in lines:
            self.words.append(line[:-1])

    def find_anagrams(self):
        self.anag = {}
        for word in self.words:
            temp = word.lower()
            word = list(word.lower())
            word.sort()
            key = ''.join(word)
            if key not in self.anag:
                self.anag[key] = [temp]
            else:
                value = self.anag[key]
                if temp in value:
                    continue
                value.append(temp)
                self.anag[key] = value
        self.sorted_anag = sorted(self.anag.items(), key=lambda x: len(x[1][0]), reverse=True)

    def n_grams(self, n=1):
        self.n_grams = []
        for pair in self.sorted_anag:
            grams = pair[1]
            if len(grams) > 1:
                if len(grams[0]) < n:
                    continue
                line = []
                for gram in grams:
                    line.append(gram)
                self.n_grams.append(line)

    def sort_anagramgs(self, f_out):
        s = sorted(self.n_grams, key=lambda x: ( len(x), len(x[0]),\
            -sum(overlap_score(x[i],x[j]) for i,j in zip(range(len(x)-1),range(1,len(x)))) ), reverse=True)
        #-sum(overlap_score(x[i],x[j]) for i,j in zip(range(len(x)-1),range(1,len(x)))) 
        f = open(f_out,'w')
        for line in s:
            f.write(str(line) + '\n')
        f.close()
        return


def overlap_score(w1, w2):
    p1 = Set()
    p2 = Set()
    i = 0
    j = 2
    while j <= len(w1):
        p1.add(w1[i:j])
        i += 1
        j += 1
    k = 0
    l = 2
    while l <= len(w2):
        p2.add(w2[k:l])
        k += 1
        l += 1
    score = len(p1 & p2)
    return score


def main():
    anagram = Anagram('dict.txt')
    anagram.find_anagrams()
    anagram.n_grams(6)
    anagram.sort_anagramgs('out.txt')

    print "Testing overlap score:"
    print "integrals : triangles"
    print overlap_score("integrals","triangles")
    print "integrals : integrals"
    print overlap_score("integrals","integrals")

if __name__ == '__main__':
    main()
