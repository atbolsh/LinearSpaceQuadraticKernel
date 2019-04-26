from project import *
from scipy.linalg import solve

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


def vector(G, v=None, lr=0.01, maxIter = 1000000, cutoff=1e-8):
   """Find the vector from projecting the ones vector to the nullspace of the error matrix
   interpreted as a metric tensor. Due to possible degeneracies, we will likely have to do
   postprocessing, but that is best reserved for another method."""
   N = G.shape[0]
   H2 = graphToH(G)
   M = getM(H2)                 #Glorious metric tensor
   ls, A = eig(M)
   A = A.T
   u = np.matmul(A, np.ones(N)) #translate L1 norm into new coordinates. Luckily they are orthogonal.
   c = converge(ls, u, v, lr, maxIter, cutoff)  #Compute projection onto null space intersected with unit sphere.
   v = solve(A, c)          #Return to the land of the living.
   return v

