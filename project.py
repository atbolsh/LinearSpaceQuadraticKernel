"""Module for projecting onto the kernel of a diagonal Mikowski metric tensor."""

import math
from copy import deepcopy

import numpy as np
from numpy.linalg import eig
from numpy.linalg import inv
from numpy.linalg import svd
from numpy.linalg import norm

import matplotlib.pyplot as plt

def ReLu(x):
   return np.maximum(0, x)

def kp(ls, v):
   """
   This is the initial projection step, before gradient descent.

   ls stands for the eigenvalues (main diagonal) of
   metric tensor M. v is the vecor we are projecting onto
   the intersection of the kernel of l and the unit sphere.
   
   That is, we find the vector x such that 
      -- norm(x) == 1
      -- x.T M x == 0
      -- the distance between x and v is minimal.
   """
   i = np.argmax(ls)
   j = np.argmin(ls)

   s  = np.sign(ls)         #All chopping must take place before this procedure.
   vp = v*ReLu(s)           #Positive space
   vn = v*ReLu(0 - s)       #Negative space
   v0 = v*(1 - np.fabs(s))  #zero space (linear kernel)
   
   vp += v0     #Might change later; what to do with the linear kernel is a good question.
   
   Rp = np.sum(vp*vp)        #L2 norms
   Rn = np.sum(vn*vn)        #In the application, this might often be 0. 


   #In case one of the projections is 0, we need this initialization.
   #I hope that it will still lead to the correct minimum.
   if Rn < 0.01/len(ls):
      vn = np.zeros(len(ls))
      vn[j] = 1.0
      Rn = 1.0
   
   if Rp < 0.01/len(ls):
      vp = np.zeros(len(ls))
      vp[i] = 1.0
      Rp = 1.0

   Qp = np.sum(vp*ls*vp)     #Minkowski norms
   Qn = 0 - np.sum(vn*ls*vn)
   
   #Now, we solve for the right multipliers on the two spaces.
   u = Rp*Qn + Rn*Qp
   a2 = Qn/u
   b2 = Qp/u
   
   
   return np.sqrt(a2)*vp + np.sqrt(b2)*vn


def project(x, u):
   """Projects u onto x"""
   return x*np.dot(x, u)/np.dot(x, x)


def project2(x, y, u):
   """Projects vector u onto the space spanned by x and y"""
   z = y - project(x, y)                #Switch to orthogonal basis.
   return project(x, u) + project(z, u)


def converge(ls, u, v=None, lr=0.01, maxIter = 1000000, cutoff=1e-8):
   """Gradient-descent based convergence.
   Uses kp to get in the correct basin of attraction."""
   if type(v) == type(None):
      v = kp(ls, u)
   s = u/norm(u)
   for i in range(maxIter):
      g = s - project2(ls*v, v, s) #Find lagrange multipliers and remove them, basically. Constrained gradient.
      if norm(g) < cutoff:
         print("Success!")
         break
      v += lr*g
      v = kp(ls, v) #To avoid accumulated errors
   return v










