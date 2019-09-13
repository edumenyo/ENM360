#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 17:25:35 2018

@author: paris
"""

import numpy as np
import matplotlib.pyplot as plt
from types import LambdaType

class LagrangeInterpolant():
        
    coefficients = {}
    
    def interpolate(self, x_star=None, x=None, y=None, func=None, num=17):

        # the case where training data is provided
        if (func is None and x_star is not None 
            and x is not None and y is not None):
            
            self.coefficients = self.computeLagrangePoly(x.flatten(), 
                                                         y.flatten())

        # the case where an exact function is provided
        elif ((x is None or y is None) and func is not None and
              isinstance(func, LambdaType)):
            if (x_star is None):
                print("No x_star was passed!")
                x_star = np.linspace(-7, 7, num=17)

            y_star = np.array([func(p) for p in x_star])
            self.coefficients = self.computeLagrangePoly(x_star.flatten(), 
                                                         y_star.flatten())
            
        # any cases where invalid paramaters are provided to interpolate
        else:
            raise ValueError()
            
        
        self.y_pred = np.polynomial.polynomial.polyval(x_star.flatten(), 
                                                       self.coefficients)
        return np.copy(self.y_pred)
        
    def createNodes(self, start, stop, n=16, intpolMethod=0):
        '''
        Creates a numpy array of nodes for use in Lagrange interpolation (with
        the interpolate() method. In the fashion of numpy.linspace(), num nodes
        are created ranging from start to stop; the nodes are either linearly
        spaced (when intpolMethod=0), are Chebyshev-Gauss-Legendre nodes (when
        intpolMethod=1), or are Chebyshev-Gauss nodes (when intpolMethod=2).
        '''
        
        if ((start is not None and stop is not None and n > 0) and
            intpolMethod in [0, 1, 2]):
            
            nodes = np.empty(n+1)
            
            # linearly-spaced nodes
            if intpolMethod == 0:
                nodes = np.linspace(start, stop, num=n+1)
            
            # Chebyshev-Gauss-Legendre nodes
            elif intpolMethod == 1:
                for i in range(n+1):
                    x_i = ((start + stop)/2 + 
                           (stop-start)/2*(-np.cos(np.pi * i/n)))
                    nodes[i] = x_i
           
            # Chebyshev-Gauss nodes
            else:
                for i in range(n+1):
                    x_i = ((start + stop)/2 - 
                           (stop-start)/2*(np.cos((2*i+1)/(n+2)*np.pi/2)))
                    nodes[i] = x_i
                
            if nodes.size != n+1:
                print("Size of nodes: ", nodes.size)
                raise Exception("Nodes either not computed or added correctly")
            return nodes
        else:
            raise ValueError()
    
    def computeLagrangePoly(self, x_star, y_star):
        npp = np.polynomial.polynomial
        
        f = np.zeros(x_star.size + 1)
        phi = np.zeros((x_star.size, f.size))
        x_var = np.copy(f)
        x_var[1] = 1
        manip_xvar = np.copy(x_var)
        for k in range(x_star.size):
            temp_phi = np.zeros(x_star.size + 1)
            temp_phi[0] = 1
            for j in range(x_star.size):
                if j == k:
                    continue
                manip_xvar = np.zeros(x_star.size + 1)
                manip_xvar[1] = 1
                manip_xvar[0] = -1*x_star[j] 
                temp = x_star[k] - x_star[j]
                temp_phi2 = npp.polymul(temp_phi, (manip_xvar / temp))
                temp_phi[:temp_phi2.size] = temp_phi2
                if temp_phi.size != f.size:
                    raise ValueError()
            phi[k] = temp_phi     
            
            f += phi[k]*y_star[k]
        return f
    
    def computeError(self, x_star, func):
#        raise NotImplementedError()
        
        if (x_star is None or func is None or 
            not isinstance(func, LambdaType)):
            raise ValueError()
        else:
            y_star = np.array([func(x_i) for x_i in x_star])
            temp_f = np.polynomial.polynomial.polyfit(x_star, y_star, 2)
            f = np.zeros(self.coefficients.size)
            f[:temp_f.size] = temp_f
            if f.size != self.coefficients.size:
                print("exact coeffs: {}, estimated coeffs: {}".format(f, 
                      self.coefficients))
                raise ValueError()
            if np.array_equal(f, self.coefficients):
                return 0
            error = np.subtract(f, self.coefficients)
            error = np.linalg.norm(error)
            error /= np.linalg.norm(f)
            return error
    
if __name__ == '__main__':
        
    # Create a function computing the Lagrange 
    # interpolating polynomial
        
    # Load training data
    training_data = np.loadtxt('HW2_training_data.txt')
    x_t = training_data[:,0:1]
    y_t = training_data[:,1:2]
    
    # Load validation data
    validation_data = np.loadtxt('HW2_validation_data.txt')
    x_v = validation_data[:,0:1]
    y_v = validation_data[:,1:2]
    
    # Create a grid to plot the predicted function
    x_s = np.linspace(x_v.min(0), x_v.max(0), 100)
    
    # Construct the Lagrange interpolant
    y_pred = LagrangeInterpolant().interpolate(x_star=x_s, x=x_t, y=y_t)

    # Plot the results
    plt.figure(1)
    plt.plot(x_v, y_v,'bo', markerfacecolor = 'None', 
             label = "Validation data")
    plt.plot(x_t, y_t,'rx', label = "Training data")
    plt.plot(x_s, y_pred, 'k-', label = "Lagrange interpolant")
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend()
       