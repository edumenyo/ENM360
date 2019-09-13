# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 20:37:12 2019

@author: edumenyo
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from autograd import grad
from types import LambdaType

class CentralFiniteDiffApproximation():
    
    func = None
    x = None
    
    def approxDerivative(self, func=None, start=-1, stop=1, numPoints=0):
        '''
        Using the central finite differences approxima function approximates 
        the derivative of func along a discretization grid [start..stop] with 
        numPoints nodes.
        '''
        if (func is None or numPoints < 1 or 
            not isinstance(func, LambdaType)):
            raise ValueError()
            
        else:
            self.func = func
            h = (stop - start)/numPoints
            self.x = np.linspace(start, stop, numPoints)
            f = np.array([self.func(x_i) for x_i in self.x])
            circulantMatrix = sparse.diags([-1, 0, 1], [-1, 0, 1],
                                                 shape=(numPoints,
                                                        numPoints)).toarray()
            approxOfDeriv = circulantMatrix*f        
            approxOfDeriv[0] = self._computeEdgeApprox(True, numPoints,
                                                       h, f)
            approxOfDeriv[numPoints-1] = self._computeEdgeApprox(False, 
                                                                 numPoints, 
                                                                 h, f)
        
            error = self._findApproxError(f, approxOfDeriv)
            return (approxOfDeriv, error)
        

    def _findApproxError(self, f=None, fPrimeApprox=None):
        if (f is None or fPrimeApprox is None or
            self.func is None or not isinstance(self.func, LambdaType) or
            self.x is None):
            raise ValueError()
                
        else:
            gradFunc = grad(self.func)
#            print("gradFunc(0): ", gradFunc(7.0))
            fPrime = np.array([gradFunc(x_i) for x_i in self.x])
#            print("f:{}\nfPrime:{}".format(f, fPrime))
            if f.size != fPrime.size:
                raise ValueError()
            error = np.subtract(fPrime, fPrimeApprox)
            error = np.linalg.norm(error)
            error /= np.linalg.norm(fPrime)
        
        return error
    
    def _computeEdgeApprox(self, isFirst=True, N=0,  h=0, f=None):
        '''
        This function approximates the derivative (using CFD) for the nodes on
        the edge of the approximation. As the calculation is different for the
        first & last node in the interval, the boolean variable isFirst can be
        as an indicator to select which to compute for. Aside from that, the
        number of nodes sampled from the interval (N), the interval between
        each adjacent pair (h), and a vectorized version of the function whose
        derivative needs to be approximated (f) must also be provided.
        '''
        
        if f is None or N <= 0 or h <= 0:
            raise ValueError()
            
        else:
            if isFirst:
                return ((0.5/h)*(-3*f[0] + 4*f[1] - f[2]))
                
            else:
                return ((0.5/h)*(3*f[N-1] - 4*f[N-2] + f[N-3]))
            
if __name__ == "__main__":
    '''
    Using the central finite differences approach to approximate the derivative
    of ifunc, I have found the convergence follows the sqrt function, meaning that
    increasing N creates much better derivative approximations at first, and after
    some time, the effect of increasing the number of equidistant points is minimal.
    '''
    ifunc = lambda j: (j+1)*np.sin(4*np.pi*j)
    
    cfdApp = CentralFiniteDiffApproximation()
    deriv10, error10 = cfdApp.approxDerivative(ifunc, numPoints=10)
    deriv20, error20 = cfdApp.approxDerivative(ifunc, numPoints=20)
    deriv50, error50 = cfdApp.approxDerivative(ifunc, numPoints=50)
    deriv100, error100 = cfdApp.approxDerivative(ifunc, numPoints=100)
    deriv200, error200 = cfdApp.approxDerivative(ifunc, numPoints=200)
    deriv500, error500 = cfdApp.approxDerivative(ifunc, numPoints=500)
    deriv1000, error1000 = cfdApp.approxDerivative(ifunc, numPoints=1000)
    
    plt.figure(1)
    plt.plot(np.array([10, 20, 50, 100, 200, 500, 1000]), 
             np.array([error10, error20, error50, error100, error200, error500,
                       error1000]),
             "bo", label="Error of central finite differences approximation")
    