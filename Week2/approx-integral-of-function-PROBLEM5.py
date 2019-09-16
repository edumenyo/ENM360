# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 00:34:24 2019

@author: edumenyo
"""

import matplotlib.pyplot as plt
import numpy as np
from types import LambdaType
from scipy import special

def main():
    ifunc = lambda j: (j+1)*np.sin(2*np.pi*j)*special.jv(2.5, j)

    ntegApp = IntegralApproximation()
    ntegApp.exactIntegral = ntegApp.approxIntegralWithGaussLegendre(ifunc, 0, 4.0,
                                                                    numPoints=200)[0]
    nteg5 = ntegApp.approxIntegral(ifunc, numPoints=5)
    nteg10 = ntegApp.approxIntegral(ifunc, numPoints=10)
    nteg15 = ntegApp.approxIntegral(ifunc, numPoints=15)
    nteg20 = ntegApp.approxIntegral(ifunc, numPoints=20)
    nteg40 = ntegApp.approxIntegral(ifunc, numPoints=40)
    nteg60 = ntegApp.approxIntegral(ifunc, numPoints=60)
    nteg80 = ntegApp.approxIntegral(ifunc, numPoints=80)
    nteg100 = ntegApp.approxIntegral(ifunc, numPoints=100)
    
    plt.figure(1)
    x_s = np.array([5, 10, 15, 20, 40, 60, 80, 100])
    trapError = [nteg5[0][1], nteg10[0][1], nteg15[0][1], nteg20[0][1], 
                 nteg40[0][1], nteg60[0][1], nteg80[0][1], nteg100[0][1]]
    glError = [nteg5[1][1], nteg10[1][1], nteg15[1][1], nteg20[1][1], 
               nteg40[1][1], nteg60[1][1], nteg80[1][1], nteg100[1][1]]
    
    plt.plot(x_s, np.array(trapError), '-o', label="Trapezoid rule")
    plt.plot(x_s, np.array(glError), '-o', label="Gauss-Legendre quadrature")
    
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()

class IntegralApproximation():
    exactIntegral = 0
    
    def approxIntegral(self, func=None, start=0, stop=4.0, 
                       numPoints=0):
        if (func is None or numPoints < 1 or not isinstance(func, LambdaType)):
            raise ValueError()
        else:
            x = np.linspace(start, stop, numPoints)
            
            # uses the trapezoid rule to approximate the integral
            trapNtegResults = self.approxIntegralWithTrapezoidRule(func, x, 
                                                                   numPoints)
                        
            #  uses the Gauss-Legendre quadrature to approximate the integral
            glNtegResults = self.approxIntegralWithGaussLegendre(func, start,
                                                                 stop,
                                                                 numPoints)
                            
            return (trapNtegResults, glNtegResults)

    def approxIntegralWithTrapezoidRule(self, func=None, linspace=None, 
                                        numPoints=0):
        '''
        Uses the trapezoid rule to sample
        '''
        if (func is None or linspace is None or linspace.size != numPoints or
            not isinstance(func, LambdaType)):
            raise ValueError()
            
        else:
            factor = (linspace[1] - linspace[0])
            integral = np.float64(0)
            for i in range(1, numPoints):
                integral += func(linspace[i])*factor
            integral += factor*0.5*(func(linspace[numPoints-1]) + 
                                    func(linspace[0]))

            error = self._findApproxAbsoluteRelativeError(self.exactIntegral, 
                                                          integral)
            return integral, error

    def approxIntegralWithGaussLegendre(self, func=None, start=0, 
                                        stop=4, numPoints=0):
        '''
        Uses the Gauss-Legendre quadrature to sample
        '''
        if (func is None or numPoints <= 0 or start == stop or
            not isinstance(func, LambdaType)):
            raise ValueError()
            
        else:
            factor = 0.5*(stop - start)
            nodes, weights = np.polynomial.legendre.leggauss(numPoints)
            nteg = np.float64(0)
            for i in range(nodes.size):
                x_i = (stop + start)/2 + (stop - start)/2*nodes[i]
                nteg += weights[i] * func(x_i) 
            
            nteg *= factor
            error = self._findApproxAbsoluteRelativeError(self.exactIntegral, 
                                                          nteg)
            return nteg, error
    
    def _findApproxAbsoluteRelativeError(self, f=None, fNtegApprox=None):
#        print("f_exact: {}, f_approx: {}".format(f, fNtegApprox))
        if f == fNtegApprox or f == 0:
            return 0
        return (f - fNtegApprox)/f
        
if __name__ == "__main__":
    main()

