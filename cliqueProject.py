import math
from copy import deepcopy

import numpy as np
from numpy.linalg import eig
from numpy.linalg import inv

import matplotlib.pyplot as plt


#I know it's slow as written, but this is python anyway.
#
#I want practice manipulating large matrices as objects, instead of writing
#spaghetti-code for loops.
#


# In order to realize the Tr(X, H) = k(k-1) condition, I set the top
# k(k-1) off-diagonal elements to 1, and the rest to 0.

def cliqueProject(A, k, H):
    """Clique projection for that question.
    Assumes H represents the edges of a graph, with 0s along
    the main diagonal."""
    B = deepcopy(A)
    s = B.shape
    d = np.diagonal(A)
    #First, make the top k diagonal elements into 1s, and the rest 0
    ind = np.argsort(d)
    rev = np.argsort(ind)
    #Switch to ordered.
    d = d[ind]
    d[0-k:] = 1
    d[:0-k] = 0
    #Switch back
    d = d[rev]
    # Now, do the projection on the other indices.
    B = B*H #Element-wise. This is a mask.
    #Switch to a line
    B = B.reshape(s[0]*s[1])
    ind = np.argsort(B)
    rev = np.argsort(ind)
    #Switch to ordered
    B = B[ind]
    B[0-k*(k-1):] = 1
    B[:0-k*(k-1)] = 0
    #Switch back to original order
    B = B[rev]
    #Switch back to square form
    B = B.reshape(s)
    #Add in the diagonal contributions
    #Again, the multiplication is element-wise. This is a mask.
    B = B + np.identity(s[0])*d
    return B


def extractClique(A, acc=8):
    """Gets indices of clique from solution matrix"""
    d = np.round(np.diagonal(A), acc)
    r = []
    for i in range(len(d)):
        if d[i] > 0.1:
            r.append(i)
    return np.array(r)


def verifyClique(c, G):
    r = 1.0
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            r = r*G[c[i], c[j]]
    if r > 0.1:
        return True
    else:
        return False


