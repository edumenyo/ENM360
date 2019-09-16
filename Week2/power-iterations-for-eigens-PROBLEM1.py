#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 18:40:27 2019

@author: paris
"""

import numpy as np
import matplotlib.pyplot as plt

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
    
    def computePowerIteration(mat, eigenPair):
        '''
        Given a random symmetric positive definite matrix (mat), this function
        calculates the next iteration of eigenPair = (eigenvector, eigenvalue),
        using the power method --- a formula that also requires the eigenPair
        from the previous iteration.
        '''
        eigenvect = eigenPair[0]
        tempProduct = np.dot(mat, eigenPair[0])
        newEigenvector = tempProduct / np.linalg.norm(tempProduct)
        newEigenvalue = (np.dot(eigenvect.T, tempProduct) / 
                  np.dot(eigenvect.T, eigenvect))
        
        return (newEigenvector, newEigenvalue.flatten())    
    
    def isEigenVerified(correctEigenvector, correctEigenvalue, 
                        currEigenvector, currEigenvalue):
        corrEig = np.round(correctEigenvalue, 4)
        currEig = np.round(currEigenvalue, 4)
        if not corrEig == currEig:
            print("expected: {}, actual: {}".format(correctEigenvalue, 
                                                    currEigenvalue))
            
            return False
        
        return True
    
    # Matrix size
    N = 200 
    
    # Construct a random symmetric positive definite matrix
    A = sample_SPD_matrix(N)
    
    # (vector [numpy array], eigenval, norm of eigenvec)
    maxEigenPair = (np.random.randn(N), np.float64(1.000)) 
    
    # same format as the above
    minEigenPair = (np.random.randn(N), np.float64(1.000))  # must manipulate to make this calc the min eigens
    xAxis = np.linspace(1, 100000, 100000)
    MaxEigenvalueApproxes = np.zeros(1000)
    MinEigenvalueApproxes = np.zeros(100000)
    
    
    for x in range(1000):
        # compute next iteration of the power method
        maxEigenPair = computePowerIteration(A, maxEigenPair)
        MaxEigenvalueApproxes[x] = maxEigenPair[1].flatten()
        
        
    alterFactor = maxEigenPair[1][0] * ((np.identity(N)))
    B = A - alterFactor 
        
    for x in range(50000):
        # compute next iteration of the power method
        minEigenPair = computePowerIteration(B, minEigenPair)
        MinEigenvalueApproxes[x] = minEigenPair[1].flatten()
        
    # correct for alteration
    minEigenPair = (minEigenPair[0], minEigenPair[1] + maxEigenPair[1])
    
    # plot the predicted eigenvalues for each iteration
    plt.figure(1)
    plt.plot(xAxis, MinEigenvalueApproxes, label="min eigenvalues")
    plt.plot(xAxis[:1000], MaxEigenvalueApproxes, label="max eigenvalues")
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend()
    
    # Compute the condition number of A
    c = np.linalg.cond(A)
    
    # Compute the eigenvalues and eigenvectors of A
    w, v = np.linalg.eig(A)
    
    # Print the condition number and the maximum eigenvalue of A
    print('Condition number of A: %e' % (c))
    print('Largest eigenvalue A (eig): %e' % (w.max()))
    print('Smallest eigenvalue A (eig): %e' % (w.min()))
    
    
    maxIndex = np.argmax(w)
    minIndex = np.argmin(w)
    
    if isEigenVerified(v[:, maxIndex], w.max(), maxEigenPair[0], maxEigenPair[1]):
        print("\nMax eigen success!")
    if isEigenVerified(v[:, minIndex], w.min(), minEigenPair[0], minEigenPair[1]):
        print("Min eigen success!")
