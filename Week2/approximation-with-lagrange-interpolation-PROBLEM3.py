# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:03:43 2019

3.) Use your Lagrange interpolation code to approximate the function 
    f(x)=2/(5+x2) in the interval x∈[−7,7] using a uniform grid consisting of
    N=13 equidistant nodes. Plot the exact function and the Lagrange 
    interpolant and comment on the accuracy of the approximation. Check if the 
    approximation can be improved by using a non-uniform grid of the same size
    corresponding to (a) the Chebyshev-Gauss-Legendre, and (b) the 
    Chebyshev-Gauss nodes.

@author: edumenyo
"""

import numpy as np
import matplotlib.pyplot as plt
lagr = __import__("lagrange-interpolations-PROBLEM2")

def main():
    '''
    Uses the Lagrange interpolation to approximate the function f(x)=2/(5+x2) 
    in the interval x∈[−7,7]. This interpolation is performed with linearly
    spaced nodes, Chebyshev-Gauss-Legendre nodes, and Chebyshev-Gauss nodes, 
    plotting the results of each (along with the graph of f). 
    
    The error of each
    method is subsequently calculated, and the program prints out the results,
    explicitly revealing the most accurate interpolation method.
    
    The best way to use this is to compare the plot of the exact function to
    the plot of each interpolation method --- individually. Upon doing so, it
    becomes clear that the ___ method is the best.
    '''
    lit = lagr.LagrangeInterpolant()
    f = lambda l: 2 / (5 + l**2)
    
    # interpolationwith linearly spaced nodes
    x_s = np.linspace(-7, 7, num=17)
    y_pred = lit.interpolate(x_star=x_s, func=f)
#    linError = lit.computeError(x_s, f)
#    print("Error with linearly-spaced nodes: {}".format(linError))
    
    plt.figure(1)
    plt.plot(x_s, np.array([f(p) for p in x_s]), 
             'k-', linewidth=7, linestyle='dashed', label="Exact function")
    plt.plot(x_s, y_pred, 'g-', label="Linspace interpolant")
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend(loc='lower center')
    
    # interpolation with Chebyshev-Gauss nodes
    plt.figure(2)
    x_s = lit.createNodes(-7, 7, 16, 2)
    y_pred = lit.interpolate(x_star=x_s, func=f)
    plt.plot(x_s, y_pred, 'b-.', label="Chebyshev-Gauss interpolant")
#    cgError = lit.computeError(x_s, f)
#    print("Error of Chebyshev-Gauss function approximation: {}".format(cgError))
    x_s = np.linspace(-7, 7, num=17)
    plt.plot(x_s, np.array([f(p) for p in x_s]), 
             'k-', linewidth=7, linestyle='dashed', label="Exact function")
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend(loc='lower center')
    
    # interpolation with Chebyshev-Gauss-Legendre nodes
    plt.figure(3)
    x_s = lit.createNodes(-7, 7, 16, 1)
    y_pred = lit.interpolate(x_star=x_s, func=f)
#    cglError = lit.computeError(x_s, f)
#    print("Error of Chebyshev-Gauss-Legendre function approximation: {}".format(cglError))
    plt.plot(x_s, y_pred, 'r-', label="Chebyshev-Gauss-Legendre interpolant")
    x_s = np.linspace(-7, 7, num=17)
    plt.plot(x_s, np.array([f(p) for p in x_s]), 
             'k-', linewidth=7, linestyle='dashed', label="Exact function")
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend(loc='lower center')

#    print(np.argmin(np.array([linError, cglError, cgError])))

if __name__ == "__main__":
    main()
