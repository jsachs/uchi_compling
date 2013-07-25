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
        if round(sum(self.pi),3) != 1.0:
            return False

        # Transition probabilities
        for i in range(self.N):
            if round(sum(self.A[i]),3) != 1.0:
                return False

        # Emission probabilities
        for i in range(self.N):
            if round(sum(self.B[i]),3) != 1.0:
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
        for s in self.X:
            alpha[(s, 0)] = self.pi[self.X_ind[s]]

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
        for s in self.X:
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
        P = sum(alpha[(i,len(word))] for i in self.X) # P(O)

        for l,t in zip(word,range(len(word))):
            for i in self.X:
                for j in self.X:
                    p[(t, i, j)] = (alpha[(i,t)] * \
                                   self.A[self.X_ind[i]][self.X_ind[j]] * \
                                   self.B[self.X_ind[i]][self.O_ind[l]] * \
                                   beta[(j,t+1)]) / P
        return p

    """
    Gammas
    calc_gammas

    Finds the gamma_i(t) quantity, used for maximization.

    Parameters:
        word: the word for which we are finding gammas.

    Returns:
        gamma: a dictionary of gamma values keyed by (i,t)
    """
    def calc_gammas(self, word):
        alpha = self.calc_forward(word)
        beta = self.calc_backward(word)
        gamma = {}

        for t in range(len(word)):
            for i in self.X:
                num = alpha[(i,t)]*beta[(i,t)]
                denom = 0
                for j in self.X:
                    denom += alpha[(j,t)]*beta[(j,t)]
                gamma[(i,t)] = num/denom
        return gamma

    """
    Expected Transitions
    expect_trans

    Finds the expected transition vector and matrices for
    the initial distribution, the state transition matrix,
    and the emission matrix.

    Parameters:
        corpus: the corpus being used as data

    Returns:
        (A, B, pi): A tuple of the updated probability matrices
    """
    def expect_trans(self, corpus):
        pi_tot = {}
        A_tot  = {}
        B_tot  = {}
        
        # get the expectations of pi(i), a(i,j), and b(i,j) for each word
        # these are then kept in a running total, and eventually normalized
        for word in corpus:
            pi = {}
            A  = {}
            B  = {}
            gamma = self.calc_gammas(word)
            p     = self.calc_expected(word)

            # calculate pi
            for i in self.X:
                pi[i] = gamma[(i,0)]
            for key, value in pi.iteritems():
                if key in pi_tot:
                    pi_tot[key] += value
                else:
                    pi_tot[key] = value

            # calculate A
            for i in self.X:
                for j in self.X:
                    num = sum(p[(t,i,j)] for t in range(len(word)))
                    denom = sum(gamma[(i,t)] for t in range(len(word)))
                    A[(i,j)] = num/denom
            for key, value in A.iteritems():
                if key in A_tot:
                    A_tot[key] += value
                else:
                    A_tot[key] = value

            # calculate B
            for i in self.X:
                for j in self.X:
                    denom = sum(p[(t,i,j)] for t in range(len(word)))
                    for t in range(len(word)):
                        num = 0
                        for t_prime in range(len(word)):
                            if word[t] == word[t_prime]:
                                num += p[(t_prime,i,j)]
                        key = (i,word[t])
                        if key in B:
                            B[key] += num/denom
                        else:
                            B[key] = num/denom
            for key, value in B.iteritems():
                if key in B_tot:
                    B_tot[key] += value
                else:
                    B_tot[key] = value

        # unpack the dictionaries into new probability matrices
        pi_f = self.pi
        A_f  = self.A
        B_f  = self.B
        for i in self.X:
            pi_f[self.X_ind[i]] = pi_tot[i]
            for j in self.X:
                A_f[self.X_ind[i]][self.X_ind[j]] = A_tot[(i,j)]
            for t in self.O:
                B_f[self.X_ind[i]][self.O_ind[t]] = B_tot[(i,t)]

        # normalize
        pi_f = normalize(pi_f)
        for i in range(self.N):
            A_f[i] = normalize(A_f[i])
        for i in range(self.N):
            B_f[i] = normalize(B_f[i])

        return (A_f, B_f, pi_f)

    """
    Forward-Backward Algorithm
    forward_backward

    Runs the FB algorithm over the entire corpus,
    attempting to classify letters as vowels or consonants.

    Parameters:
        corpus: our test corpus

    Returns:

    """
    def forward_backward(self, corpus):
        flag = True
        x = 0
        while flag and x < 50:
            check = []
            for i in range(self.N):
                check.append(range(self.T))
            for i in range(self.N):
                for j in range(self.T):
                    check[i][j] = self.B[i][j]

            self.A, self.B, self.pi = self.expect_trans(corpus)
            if self.sanity_check() == False:
                print 'Probability error'
                return 0

            # Closeness condition
            x += 1
            flag = False
            for i in range(self.N):
                for j in range(self.T):
                    if abs((check[i][j] - self.B[i][j])/check[i][j]) > .1:
                        flag = True

        print self.pi
        print self.A
        
        print 'letter'.rjust(10), 'prob'.rjust(20), 'state'.rjust(30)
        for state in self.X:
            for letter in self.O:
                print letter.rjust(10), repr(round(self.B[self.X_ind[state]][self.O_ind[letter]],5)).rjust(20), state.rjust(30)


"""
Vector Normalization
normalize

Helper function to normalize vectors when
expectation values are recalculated.

Parameters:
    vector: the vector, in the form of a list

Returns:
    norm: a python list, and the normalized vector
"""
def normalize(vector):
    denom = sum(vector)
    norm = vector
    for i in range(len(vector)):
        norm[i] = vector[i]/denom
    return norm

# ================================================================================

def main():

    # ======== 2-State HMM ========

    state_list = ['S1', 'S2']
    out_list   = ['a', 'b', 'c', 'd', 'e',
                  'f', 'g', 'h', 'i', 'j',
                  'k', 'l', 'm', 'n', 'o',
                  'p', 'q', 'r', 's', 't',
                  'u', 'v', 'w', 'x', 'y', 'z']

    init_vec  =  [0.75, 0.25]
    trans_mat = [[0.25, 0.75],
                 [0.75, 0.25]]
    out_mat   = [[0.17, 0.33, 0.33, 0.17],
                 [0.33, 0.17, 0.17, 0.33]]

    hmm = HMM(state_list, out_list, init_vec, trans_mat)
    for i in state_list:
        for j in out_list:
            hmm.assign_out_prob(i, j, 1.0/len(out_list))

    if hmm.sanity_check() == False:
        print "error: hmm fails sanity check\n"
        exit(0)

    lines = open('brown_1000.txt','r').readlines()
    corpus = []
    for line in lines:
        corpus.append(line[:-1])

    hmm.forward_backward(corpus)

    # ======== 3-State HMM ========

    state_list2 = ['S1', 'S2', 'S3']
    init_vec2 = [0.5, 0.25, 0.25]
    trans_mat2 = [[0.5, 0.25, 0.25],
                  [0.5, 0.25, 0.25],
                  [0.5, 0.25, 0.25]]

    hmm2 = HMM(state_list2, out_list, init_vec2, trans_mat2)
    for i in state_list2:
        for j in out_list:
            hmm2.assign_out_prob(i, j, 1.0/len(out_list))

    if hmm2.sanity_check() == False:
        print "error: hmm fails sanity check\n"
        exit(0)

    hmm2.forward_backward(corpus)

    return 0

if __name__ == '__main__':
    main()

