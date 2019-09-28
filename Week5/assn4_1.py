#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 20:12:49 2019

@author: paris
"""

import numpy as np
import matplotlib.pyplot as plt

from models import LinearSystem
from optimizers import SGD

np.random.seed(1234)
    
if __name__ == "__main__":
    
    def gen_matrix(N, CondNum = 10.0):
        A = np.random.randn(N,N)
        U,S,V = np.linalg.svd(A)
        S[S!=0] = np.linspace(CondNum,1,N)
        A = np.matmul(U, np.matmul(np.diag(S), V.T))
        return A
        
    # Generate data
    N = 512
    CondNum = 10.0
    A = gen_matrix(N, CondNum)
    b = np.random.randn(N,1)
    x_star = np.linalg.solve(A,b)
    
    # Define optimizer
    optimizer = SGD(N, lr = 1e-3)
    
    # Define model
    x0 = np.random.randn(N,1)
    model = LinearSystem(A, b, x0)
    
    # Solve
    x_pred = model.solve(optimizer, batch_size = 32, tol = 1e-5)
    
    # Print error
    error = np.linalg.norm(x_star - x_pred, 2)/np.linalg.norm(x_star, 2)
    print('Relative error: %e' % (error))
    
    # Plot
    plt.figure()
    plt.plot(x_star, x_star, 'k--', linewidth = 1, alpha = 0.5)
    plt.plot(x_star, x_pred, 'ko')
    plt.xlabel('Exact')
    plt.ylabel('Prediction')
          
    plt.figure()
    plt.plot(model.loss_log, 'k')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    
    