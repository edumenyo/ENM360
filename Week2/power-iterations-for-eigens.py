# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 17:20:22 2019

1.) The power iteration (also known as the power method) is an eigenvalue 
algorithm:
    given a diagonalizable matrix A, the algorithm will produce a number λ, 
    which is the greatest (in absolute value) eigenvalue of A, and a nonzero 
    vector v, which is a corresponding eigenvector of λ, that is, Av=λv. 
Implement a Python class for performing 100 power iterations to find the 
    largest and the smallest eigevalue of a random symmetric matrix A, and its 
    corresponding eignevectors. Plot the predicted eigvanlue in every iteration
    of the algorithm and verify the accuracy of your results using NumPy’s 
    built-in function eig function.

@author: edumenyo
"""

import numpy as np
import random as pyrand

class RandomPowerIterationGenerator:      
    @staticmethod
    def run():

        # (vector [numpy array], eigenval, norm of eigenvec)
        maxEigens = ({}, 0, 0) 
        
        # same format as the above
        minEigens = ({}, 0, 0) 
        currEigens = ({}, 0, 0)
        
        RPIG = RandomPowerIterationGenerator
        np.random()
            
        # generate random matrix for running our power iterations
        try:
            mat = RPIG.__generateRandDiagMatrix(pyrand.randint(1, 500))
        except:
            raise Exception(
                    "Could not generate power iterations on random array")

        correctEigenPairsFinal = np.linalg.eig(mat)
        correctEigenPairs = np.copy(correctEigenPairsFinal)

        x = 0        
        while x < 100:
            x += 1
            currEigens = RPIG.__calcNextEigens(mat, currEigens, x)
            
            RPIG.__buildEigenPlot

            if not RPIG.__isEigenVerified(correctEigenPairs):
                raise Exception("Unknown eigenvalue or eigenvector computed")
            
            if currEigens[1] < minEigens[1]:
                minEigens = currEigens
                # replace value(s)
                
            if currEigens[1] > maxEigens[1]:
                maxEigens = currEigens
            
            # plot the predicted eigenvalue
            
       
        eigenvalExtrema = RPIG.__getEigenvalueExtrema(correctEigenPairsFinal)
        if (minEigens[1] != eigenvalExtrema[0] or
            maxEigens[1] != eigenvalExtrema[1]):
            raise Exception("Incorrect max/min eigenvalue computed")
            
        return (maxEigens[1], minEigens[1])
    
    @staticmethod
    def __calcNextEigens(mat, currEigens, x):
        '''
        This function 
        '''
        
        eigenvec = {}
        eigenval = 0
        normOfEigenvec = 0
        
        # nextEigenvec = mat # this 
        raise NotImplementedError()
#        return (eigenvec, eigenval, normOfEigenvec)
    
    @staticmethod
    def __generateRandDiagMatrix(size):
        '''
        This function generates a random symmetric, diagonalizable matrix
        rdMat, where 
        '''
        
        # old method
#        rdMat = np.random.rand(size, size)
#        rdLowerMat = np.tril(rdMat) # makes the matrix lower triangular
#        rdMat = rdLowerMat.T + rdLowerMat # makes matrix symmetric
#        
#        # check that rdMat = rdMat.T
#        if not np.array_equal(rdMat, rdMat.T):
#            raise Exception("Could not generate symmetric matrix")
            
        
        
        print("Generated random diagonalizable matrix of size {}".format(size))
        return rdMat
     
    @staticmethod
    def __buildEigenPlot(prevPlot, eigenvalue):
        raise NotImplementedError()
        
    @staticmethod
    def __getEigenvalueExtrema(correctEigenPairs):
        minEigenvalue = 0
        maxEigenvalue = 0
       
        # (just need to sort, then get the first elem (minEigenval) 
        #   & the last elem (maxEigenval)
        
        raise NotImplementedError()
#        return (minEigenvalue, maxEigenvalue)
    
    
    @staticmethod
    def __isEigenVerified(correctEigenPairs, currEigens):
        # sort eigenvalues & vectors
            # fail if not the same size
        
        # compare eigenvalues & vectors (iteratively)
        
#        if not correctEigenPairs[0].contains(currEigens[1]) and 
        raise NotImplementedError()
 

if __name__ == "__main__":             
    for i in range(50):
        (minEigenval, maxEigenval) = RandomPowerIterationGenerator.run() 
        print(
"""For iteration {}:\n\tminimum eigenvalue: {},\n\tmaximum eigenvalue: {}
""".format(i, minEigenval, maxEigenval)
        )