# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 21:10:26 2019

@author: edumenyo
"""
from models import BayesianLinearRegression
import numpy as np
import math
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.stats import norm

_basisTypes = ["identity", "monomial", "fourier", "legendre"]

class BayesianLinearRegressionFeatures:
    
    def __init__(self):
       self.basisType = None 
    
    def BuildIdentityFeaturedModel(self, X_train, Y_train, 
                                   num_features, consts=None):
        
        features = X_train
        self.basisType = "identity"
        return BayesianLinearRegression(features, Y_train, consts[0], consts[1])
    
    def FormatX_star(self, X_star, num_features):
        if self.basisType is None or self.basisType not in _basisTypes:
            raise ValueError()
            
        newMat = []
        if self.basisType == "identity":
            newMat = X_star

        elif self.basisType == "monomial":
            newMat = np.zeros((X_star.size, num_features+1))
            for i in range(X_star.size):
                for j in range(num_features+1):
                    newMat[i][j] = math.pow(X_star[i], j)
#                     print("\trow size: {}", features[i].size)

        elif self.basisType == "fourier":
            newMat = np.zeros((X_star.size, 2*(num_features+1)))
            for i in range(X_star.size):
                for j in range(num_features+1):
                    newMat[i][2*j] = np.sin(j * np.pi * X_star[i])
                    newMat[i][2*j+1] = np.cos(j * np.pi * X_star[i])
                    
        elif self.basisType == "legendre":            
            newMat = np.zeros((X_star.size, num_features+1))
            for j in range(num_features+1):
                basis = np.polynomial.Legendre.basis(j, [0, 3], [0, 3])
                for i in range(X_star.size):
                    newMat[i][j] = basis(X_star[i])
        
        return newMat
      
    def BuildMonomialFeaturedModel(self, X_train, Y_train, 
                                   num_features, consts=None):
        
        features = np.ones((X_train.size, num_features+1))
        print("monomial model: ")
        for i in range(X_train.size):
            for j in range(num_features+1):
                features[i][j] = math.pow(X_train[i], j)
#                 print("\trow size: {}", features[i].size)
#             print("\t{}".format(features[i]))
        self.basisType = "monomial"
        return BayesianLinearRegression(features, Y_train, consts[0], consts[1])
    
    def BuildFourierFeaturedModel(self, X_train, Y_train, 
                                  num_features, consts=None):
#        raise NotImplementedError()
        features = np.zeros((X_train.size, 2*(num_features+1)))
        for i in range(X_train.size):
            for j in range(num_features+1):
                features[i][2*j] = np.sin(j * np.pi * X_train[i])
                features[i][2*j+1] = np.cos(j * np.pi * X_train[i])
#                 print("\trow size: {}", features[i].size)
#             print("\t{}".format(features[i]))
        self.basisType = "fourier"
        return BayesianLinearRegression(features, Y_train, consts[0], consts[1])
    
    def BuildLegendreFeaturedModel(self, X_train, Y_train, 
                                   num_features, consts=None):
#        raise NotImplementedError()
        
        features = np.zeros((X_train.size, num_features+1))
        for j in range(num_features+1):
            basis = np.polynomial.Legendre.basis(j, [0, 3], [0, 3])
            for i in range(X_train.size):
                features[i][j] = basis(X_train[i])
                
#        basis = np.polynomial.legendre.legfit(X_train.flatten(),
#                                              Y_train.flatten(),
#                                              deg=num_features)
#        shape = basis.shape
#        features = np.polynomial.legendre.legval(X_train, basis)
#        print("Legendre legfit basis shape: {}".format(basis.shape))
        
        self.basisType = "legendre"
        return BayesianLinearRegression(features, Y_train, consts[0], consts[1])  