# ================================================================================
# Jacob Sachs
# Computational Linguistics
# Hidden Markov Models Part I
# ================================================================================

class HMM:
    """
    Initialization
    __init__

    Takes lists of states, outputs,
    and optional probabilities, and sets up
    a Hidden Markov Model instance.

    Parameters:
        state_list:         list of states in the HMM
        out_list:           list of outputs
        init_prob  (None):  initial state probabilities
        trans_prob (None):  transition probabilities
        out_prob   (None):  output probabilities

    Returns:
        None
    """
    def __init__(self, state_list, out_list,
                 init_prob = None, trans_prob = None, out_prob = None):
        # Lists of states and outputs
        self.X = state_list
        self.O = out_list
        self.N = len(state_list)
        self.T = len(out_list)

        # Vector for initial state probabilies
        self.pi = []
        for i in range(self.N):
            if init_prob:
                self.pi.append(init_prob[i])
            else:
                self.pi.append(0)

        # Matrix for transition probabilities
        self. A = []
        for i in range(self.N):
            self.A.append(range(self.N))
        for i in range(self.N):
            for j in range(self.N):
                if trans_prob:
                    self.A[i][j] = trans_prob[i][j]
                else:
                    self.A[i][j] = 0

        # Matrix for output probabilities
        self.B = []
        for i in range(self.N):
            self.B.append(range(self.T))
        for i in range(self.N):
            for j in range(self.T):
                if out_prob:
                    self.B[i][j] = out_prob[i][j]
                else:
                    self.B[i][j] = 0

        # Index dictionaries
        self.X_ind = {}
        for i in range(self.N):
            self.X_ind[self.X[i]] = i
        self.O_ind = {}
        for j in range(self.T):
            self.O_ind[self.O[j]] = j

    """
    Transtion Probability Assignment
    assign_trans_prob

    Takes an initial state, a final state, and a probability,
    and reassigns the probability of a transition between these states.

    Parameters:
        i_state: an initial state
        f_state: a final state
        prob:    the probability of transition

    Returns:
        None
    """
    def assign_trans_prob(self, i_state, f_state, prob):
        self.A[self.X_ind[i_state]][self.X_ind[f_state]] = prob

    """
    Output Probability Assignment
    assign_out_prob

    Takes a state, an output, and a probability,
    and reassigns the probability of making an output in a state.

    Parameters:
        state: the state in which an output is made
        out:   the specific output in question
        prob:  the probability of making this output

    Returns:
        None
    """
    def assign_out_prob(self, state, out, prob):
        self.B[self.X_ind[state]][self.O_ind[out]] = prob

    """
    Initial Probability Assignment
    assign_init_prob

    Assigns a new initial probability to a state.

    Parameters:
        state: the state in question
        prob:  the new initial probability for a state

    Returns:
        None
    """
    def assign_init_prob(self, state, prob):
        self.pi[self.X_ind[state]] = prob

    """
    Probability Distribution Validation
    sanity_check

    Ensures that all probability distributions are normalized.

    Parameters:
        None

    Returns:
        False: sanity check fails
        True:  else
    """
    def sanity_check(self):
        # Initial state probabilities
        if sum(self.pi) != 1.0:
            return False

        # Transition probabilities
        for i in range(self.N):
            if sum(self.A[i]) != 1.0:
                return False

        # Emission probabilities
        for i in range(self.N):
            if round(sum(self.B[i]),5) != 1.0:
                return False

        # Success
        else:
            return True

    """
    Forward Probability
    calc_forward

    Calculates the forward probability for each alpha.

    Parameters:
        word: the word on which we calculate alpha.

    Returns:
        alpha: a dictionary of alpha(i, t) values.
    """
    def calc_forward(self, word):
        alpha = {}

        # Initialize t=0 alphas
        for s in range(self.N):
            alpha[(s, 0)] = self.pi[s]

        # Induction
        for t in range(1, len(word)+1):
            for f_state in self.X:
                alpha[(f_state, t)] = 0
                for i_state in self.X:
                    alpha[(f_state, t)] += alpha[(i_state, t-1)] * \
                                           self.B[self.X_ind[i_state]][self.O_ind[word[t-1]]] * \
                                           self.A[self.X_ind[i_state]][self.X_ind[f_state]]
        return alpha

    """
    Backward Probability
    calc_backward

    Calculates the backward probability for each beta.

    Parameters:
        word: the word on which we calculate beta.

    Returns:
        beta: a dictionary of beta(i, t) values.
    """
    def calc_backward(self, word):
        beta = {}

        # Initialize t=T+1 betas
        for s in range(self.N):
            beta[(s, len(word))] = 1

        # Induction
        for t in range(len(word), 0, -1):
            for i_state in self.X:
                beta[(i_state, t-1)] = 0
                for f_state in self.X:
                    beta[(i_state, t-1)] += beta[(f_state, t)] * \
                                          self.B[self.X_ind[i_state]][self.O_ind[word[t-1]]] * \
                                          self.A[self.X_ind[i_state]][self.X_ind[f_state]]
        return beta

    """
    Expected Count
    calc_expected

    Finds the probabilities of going from i->j at time t.
    For a fixed t, summing these probabilities should equal 1.0.
    The distribution over all pairs (i,j) is the soft counts.

    Parameters:
        word: the word for which we are finding expected counts.

    Returns:
        p: a dictionary of p(t, i, j) probabilities for transition i->j at time t
    """
    def calc_expected(self, word):
        alpha = self.calc_forward(word)
        beta  = self.calc_backward(word)
        p = {}
        P = sum(alpha[(i,len(word))] for i in range(self.N)) # P(O)

        for l,t in zip(word,range(len(word))):
            for i in range(self.N):
                for j in range(self.N):
                    p[(l, i, j)] = (alpha[(i,t)] * \
                                   self.A[i][j] * self.B[i][self.O_ind[l]] * \
                                   beta[(j,t+1)]) / P
        return p


"""
Soft Count Step
soft_counts

Find the probability of transmission from i to j at time t
over the entire corpus. Then sum over the whole corpus, and
determine how often each state emitted a letter.

Parameters:
    hmm: an HMM class instance
    corpus: the corpus to run the HMM on

Returns:
    ratios: a dictionary of S1/S2 values for each letter in our corpus
"""
def soft_counts(hmm, corpus):
    p_list = []
    for word in corpus:
        p_list.append(hmm.calc_expected(word))
    ratios = {}
    for l in hmm.O:
        states = []
        for k in hmm.X:
            states.append(0)
        for i in hmm.X:
            for j in hmm.X:
                for p in p_list:
                    if (l, i, j) in p:
                        states[i] += p[(l, i, j)]
        if states[1] != 0:
            ratios[l] = states[0]/states[1]
        else:
            ratios[l] = 'NaN'
    return ratios

def main():
    S1 = 0
    S2 = 1
    state_list = [S1, S2]
    out_list   = ['a', 'b', 'c', 'd', 'e',
                  'f', 'g', 'h', 'i', 'j',
                  'k', 'l', 'm', 'n', 'o',
                  'p', 'q', 'r', 's', 't',
                  'u', 'v', 'w', 'x', 'y',
                  'z', '#']

    init_vec  =  [0.75, 0.25]
    trans_mat = [[0.25, 0.75],
                 [0.75, 0.25]]
    out_mat   = [[0.17, 0.33, 0.33, 0.17],
                 [0.33, 0.17, 0.17, 0.33]]

    hmm = HMM(state_list, out_list, init_vec, trans_mat)
    for i in range(len(state_list)):
        for j in range(len(out_list)):
            hmm.assign_out_prob(i, out_list[j], 1.0/len(out_list))

    if hmm.sanity_check() == False:
        print "error: hmm fails sanity check\n"
        exit(0)

    lines = open('brown_1000.txt','r').readlines()
    corpus = []
    for line in lines:
        corpus.append('#' + line[:-1] + '#')

    ratios = soft_counts(hmm, corpus)
    print 'letter'.rjust(10), 'S1/S2'.rjust(20)
    for key in ratios:
        print repr(key).rjust(10), repr(ratios[key]).rjust(20)

    return 0

if __name__ == '__main__':
    main()

