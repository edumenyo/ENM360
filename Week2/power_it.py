#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 18:40:27 2019

@author: paris
"""

import numpy as np

# Fix random seed for reproducible results
np.random.seed(1234)
    
if __name__ == "__main__":
    

    def sample_SPD_matrix(n):
        '''
        This function generates a random symmetric positive definite matrix as:
        A = Q^{T} D Q,
        where Q is a square matrix with entries sampled from a standard normal 
        distribution and D is a diagonal matrix with positive numbers sampled from a 
        standard normal distribution.
        '''    
        Q = np.random.randn(n,n)
        eigen_mean = 1
        D = np.diag(np.abs(eigen_mean+np.random.randn(n,1)).flatten())
        A = np.matmul(Q.T, np.matmul(D,Q))
        return A
    
    # Matrix size
    N = 200 
    
    # Construct a random symmetric positive definite matrix
    A = sample_SPD_matrix(N)
    
    # Compute the condition number of A
    c = np.linalg.cond(A)   
    
    # Compute the eigenvalues and eigenvectors of A
    w, v = np.linalg.eig(A)
    
    # Print the condition number and the maximum eigenvalue of A
    print('Condition number of A: %e' % (c))
    print('Largest eigenvalue A (eig): %e' % (w.max()))
