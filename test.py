from eigen import *
from graphReader import *
from BigC import biggestClique
from BigC import verifyClique
import sys

def graphToH(G):
   """For a connectivity matrix G, returns the corresponding H2."""
   H2 = deepcopy(G)
   np.fill_diagonal(H2, 1)
   return H2

#Gets the prototypical error matrix.
def getM(H2):
   """Matrix of all the holes"""
   #It might be useful to add random weights to these, for symmetry breaking, since we are 
   #on the kernel anyway. Will consider.
   return np.ones(H2.shape) - H2


G = getGraph()
H2 = graphToH(G)
M = getM(H2)                 #Glorious metric tensor

n, basis, codom, full, transformed = maxSpace(M)

print(basis)
print(codom)
print(full)
np.set_printoptions(threshold=sys.maxsize)
print(transformed.round(2))





