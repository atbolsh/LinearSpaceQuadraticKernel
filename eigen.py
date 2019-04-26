"""Module for building a linear subspace of a Minkowski tensor"""

import math
from copy import deepcopy

import numpy as np
from numpy.linalg import eig
from numpy.linalg import inv
from numpy.linalg import svd
from numpy.linalg import norm

import matplotlib.pyplot as plt



def ReLU(x):
   return np.maximum(0, x)

def maxSpace(M):
    """Returns a large space x, such that x M x == 0"""
    ls, A = eig(M)
#    print np.matmul(M, A[0])/A[0]
#    print(ls[0])
    i = np.argmax(ls)
    j = np.argmin(ls)

    s  = np.sign(ls)         #All chopping must take place before this procedure.
    
#    print(np.sum(ReLU(s)))
#    print(np.sum(ReLU(0 - s)))
    
    ps = []
    ns = []
    ks = []
    
    for i in range(M.shape[0]):
        if ls[i] > 0:
            ps.append(i)
        elif ls[i] < 0:
            ns.append(i)
        else:
            ks.append(i)
#    print np.min(ls[ps])
#    print np.max(ls[ns])   
    
    n = min(len(ps), len(ns)) #size of this secret space
#    print(n)
    vecs = []

    for i in range(n):
        beta = np.sqrt(0 - ls[ps[i]]/ls[ns[i]])
#        print(beta)
        x = (A[:, ps[i]] + beta*A[:, ns[i]])
        vecs.append(x/norm(x))
    
    basis = np.array(vecs)
    codom = list(np.matmul(M, basis.T).T)
#    print(type(codom))
    for i in range(n):
        codom[i] /= norm(codom[i])
    if len(ps) > n:
        codom += list(A[:, ps[n:]].T)
    if len(ns) > n:
        codom += list(A[:, ns[n:]].T)
    if len(ks) > 0:
        codom += list(A[:, ks].T)

#    print(codom)

    codom = np.array(codom)
#    print(codom.shape)
    full  = np.concatenate((basis, codom), 0)
#    print(full.shape)
    
    transformed = np.matmul(full, np.matmul(M, full.T))
        
    
#    print np.sum(np.abs(np.matmul(basis, np.matmul(M, basis.T))))
    print np.sum(np.abs(np.matmul(full.T, full) - np.eye(M.shape[0])))
    return n, basis, codom, full, transformed
    


   

