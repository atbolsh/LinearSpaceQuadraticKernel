from NP import *
from graphReader import getGraph
from cliqueProject import extractClique
from cliqueProject import verifyClique


#I know this function is inefficient, but that's not a concern.
def biggestClique(y, G):
    """Returns the biggest clique from key y, by looking at that biggest k
    elements of y until it is no longer a clique."""
    l = len(y)
    ind = np.argsort(y)
    rev = np.argsort(ind)
    R = []
    for i in range(1, l):
        k = np.zeros(l)
        k[-i:] = 1
        k = k[rev]
        r = extractClique(k)
        if verifyClique(r, G):
            R = r
        else:
            break
    return R

def verifyClique(c, G):
    r = 1.0
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            r = r*G[c[i], c[j]]
    if r > 0.1:
        return True
    else:
        return False


def extractClique(A, acc=4):
    """Gets indices of clique from solution vector"""
    d = np.round(A, acc)
    r = []
    for i in range(len(d)):
        if d[i] > 0.1:
            r.append(i)
    return np.array(r)

def extendByOne(c, G):
    N = G.shape[0]
    for i in range(N):
        if i not in c:
            if verifyClique(c + [i], G):
                return c + [i]
    print "Maximal!"
    return c

        






