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
    

#    quad = integrate.quad
#
#    print("\nresult5:\n{} ?= {}, ({})".format(nteg5[0], nteg5[1],   quad(ifunc, 0, 4)[0]))
#    print("result10:\n{} ?= {}, ({})".format(nteg10[0], nteg10[1], quad(ifunc, 0, 4)[0]))
#    print("result15:\n{} ?= {}, ({})".format(nteg15[0], nteg15[1], quad(ifunc, 0, 4)[0]))
#    print("result20:\n{} ?= {}, ({})".format(nteg20[0], nteg20[1], quad(ifunc, 0, 4)[0]))
#    print("result40:\n{} ?= {}, ({})".format(nteg40[0], nteg40[1], quad(ifunc, 0, 4)[0]))
#    print("result60:\n{} ?= {}, ({})".format(nteg60[0], nteg60[1], quad(ifunc, 0, 4)[0]))
#    print("result80:\n{} ?= {}, ({})".format(nteg80[0], nteg80[1], quad(ifunc, 0, 4)[0]))
#    print("result100:\n{} ?= {}, ({})".format(nteg100[0], nteg100[1], quad(ifunc, 0, 4)[0]))
#    print("nteg solution: {}\n{}".format(ntegSolution, quad(ifunc, 0, 4)[0]))
    
    plt.figure(1)
    x_s = np.array([5, 10, 20, 50, 100, 200, 500, 1000])
    trapError = [nteg5[0][1], nteg10[0][1], nteg15[0][1], nteg20[0][1], 
                 nteg40[0][1], nteg60[0][1], nteg80[0][1], nteg100[0][1]]
    glError = [nteg5[1][1], nteg10[1][1], nteg15[1][1], nteg20[1][1], 
               nteg40[1][1], nteg60[1][1], nteg80[1][1], nteg100[1][1]]
    
    plt.plot(x_s, np.array(trapError), '-o', label="Trapezoid rule")
    plt.plot(x_s, np.array(glError), '-o', label="Gauss-Legendre quadrature")
#    plt.plot(np.array([200]), ntegSolution[1], "-o", label="Exact integration")
    
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

            error = self._findApproxAbsoluteRelativeError(self.exactIntegral, integral)
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
            error = self._findApproxAbsoluteRelativeError(self.exactIntegral, nteg)
            return nteg, error

    
#    def _findWeight_i(self, node_j, funcVal):
#        return 2 / ((1 - node_j**2) * (funcVal**2))
#    
#    def _findNodes(self, legendreNPlus1):
#        return np.polynomial.polynomial.polyroots(legendreNPlus1)
#        
#    
#    def _calculateLegendrePoly(self, degree=-1):
#        if degree < 0:
#            raise ValueError()
#        
#        elif degree == 0:
#            return np.ones(1)
#        elif degree == 1:
#            return np.array([0, 1])
#        
#        else:
#            legendre = np.zeros((degree+2, degree+2))
#            legendre[0][0] = 1
#            legendre[1][0] = 0
#            legendre[1][1] = 1
#            npp = np.polynomial.polynomial
#            for k in range(2, degree+2):
##                print("-----\nlegendre[{}] term\n{}\n".format(k-1, legendre[k-1]))
#                newarr = npp.polymulx((2*(k-1)+1)/k*legendre[k-1])
#                np.copyto(legendre[k][0:newarr.size], newarr)
##                print("legendre[{}] term\n{}\n".format(k, legendre[k]))
#                
#                legendre[k] -= ((k-1)/(k)*legendre[k-2])
##                print("full legendre[{}] term\n{}\n".format(k, legendre[k]))
#                    
#            return legendre
    
    def _findApproxAbsoluteRelativeError(self, f=None, fNtegApprox=None):
#        print("f_exact: {}, f_approx: {}".format(f, fNtegApprox))
        if f == fNtegApprox or f == 0:
            return 0
        return (f - fNtegApprox)/f
        
if __name__ == "__main__":
    main()

