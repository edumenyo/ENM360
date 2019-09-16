#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 12:27:37 2018

@author: paris
"""

import autograd.numpy as np
from autograd import grad
from scipy.special import factorial
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    def f(x):
        return np.sin(x) + np.cos(x)
    
    def TaylorSeries(f, x, x0, n = 2):
        T = f(x0)*np.ones_like(x)
        grad_f = grad(f)
        for i in range(0, n):
            T += grad_f(x0)*(x-x0)**(i+1) / factorial(i+1)
            grad_f = grad(grad_f)
        return T 
        
    
    N = 100
    x = np.linspace(-4.0,4.0,N)
    y = f(x)
    
    x0 = 0.0
    
    n = [0, 1, 5, 10]
    plt.figure(1)
    plt.plot(x, y, 'k-', label = 'f')
    for i in range(0, len(n)):
        T = TaylorSeries(f, x, x0, n[i])
        plt.plot(x, T, '--', label = '$T_{%d}$' % (n[i]))
    plt.xlabel('$x$')
    plt.ylabel('$y$')    
    plt.legend()
    