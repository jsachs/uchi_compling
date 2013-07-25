#==============================================================================#
#
# Jacob Sachs
# String-Edit Distance Problem
# Computational Linguistics
#
#==============================================================================#

from sets import Set
import sys

"""
For convention, we will use '#' to represent the null string.
"""
vowels = Set(['a','e','i','o','u'])

"""
Cost Function

Parameters:
    - a: the first string
    - b: the second string
    (strings have len <= 1)

Returns:
    - a float value of cost
"""
def cost(a, b):
    if a == '#' and b == '#':  # Identity
        return 0
    elif a == '#' or b == '#':  # Deletion/Insertion
        return 1.0
    elif a in vowels and b in vowels:  # Vowel-vowel association
        return 0.5
    elif a in vowels or b in vowels:  # Vowel-consonant association
        return 1.2
    else:  # Consonant-consonant association
        return 0.6

"""
String Edit Distance

Parameters:
    - A: a string
    - B: a string

Returns:
    - D: a matrix of TODO what is this actually a matrix of???
"""
def edit_distance(A, B):
    D = [[0]*(len(B)+1) for i in range(len(A)+1)]

    for i in range(1, len(A)+1):
        D[i][0] = D[i-1][0] + cost(A[i-1],'#')
    for j in range(1, len(B)+1):
        D[0][j] = D[0][j-1] + cost('#', B[j-1])

    for i in range(1, len(A)+1):
        for j in range(1, len(B)+1):
            m1 = round( D[i-1][j-1] + cost(A[i-1], B[j-1]), 5)
            m2 = round( D[i-1][j]   + cost(A[i-1], '#'),    5)
            m3 = round( D[i][j-1]   + cost('#', B[j-1]),    5)
            D[i][j] = min(m1, m2, m3)

    return D

"""
Least Cost Trace Path

Parameters:
    - A: a string
    - B: a string
    - D: an output matrix of edit_distance(A, B)

Returns:
    - a list of tuples of the least cost trace from A to B, and the operation type
"""
def least_cost_trace(A, B, D):
    i = len(A)
    j = len(B)
    trace = []

    while i > 0 and j > 0:
        if D[i][j] == D[i-1][j] + cost(A[i-1], '#'):
            trace.append((i-1, '#', 'delete'))
            i -= 1
        elif D[i][j] == D[i][j-1] + cost('#', B[j-1]):
            trace.append(('#', j-1, 'insert'))
            j -= 1
        else:
            trace.append((i-1,j-1, 'change'))
            i -= 1
            j -= 1

    return trace

"""
Display Edit Sequence

Parameters:
    - A: the initial string
    - B: the final string
    - trace: the sequence of edits

Returns:
    - None
"""
def print_sequence(A, B, trace):

    trace.reverse()

    print '{0:>7} | {1:>5} | {2:>5} | {3:<20}'.format('op','cost','tot','string')
    print '-'*80
    print '{0:>7} | {1:>5} | {2:>5} | {3:<20}'.format('init','0','0', A)

    string = list(A)
    tot = 0

    for l in range(len(trace)):
        op = trace[l][2]
        a = trace[l][0]
        b = trace[l][1]
        if isinstance(a, int):
            a = A[a]
        if isinstance(b, int):
            b = B[b]
        c  = cost(a, b)

        if op == 'change':
            if string[trace[l][0]] == B[trace[l][1]]:
                continue
            else:
                string[trace[l][0]] = B[trace[l][1]]
        elif op == 'delete':
            string[trace[l][0]] = ''
        elif op == 'insert':
            string.insert(trace[l][0], B[trace[l][1]])

        tot += c
        print '{0:>7} | {1:>5} | {2:>5} | {3:<20}'.format(op,c,tot,''.join(string))

"""
Compute String-Edit Sequence

Parameters:
    - A: the initial string
    - B: the final string
    - f: a file name

Returns:
    - None
    (Writes to the listed file)
"""
def compute_sequence(A, B, f):

    sys.stdout = open(f, 'w')

    D = edit_distance(A, B)
    trace = least_cost_trace(A, B, D)

    print_sequence(A, B, trace)
    sys.stdout.close()


if __name__ == '__main__':

    A = 'thenameofthegame'
    B = 'theresmyname'
    compute_sequence(A, B, 'test3.txt')

    A = 'ninakushukuru'
    B = 'unamshukuru'
    compute_sequence(A, B, 'test4.txt')



