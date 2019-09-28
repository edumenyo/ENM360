#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 10:35:11 2018

@author: paris
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.stats import norm

from models import BayesianLinearRegression
from features import BayesianLinearRegressionFeatures

if __name__ == "__main__": 
    
    # N is the number of training points.
    N = 500
    M = 1
    noise_var = 0.3
    alpha = 1.0/noise_var
    beta = 0.2
    
    # Create random input and output data
    X = lhs(1, N)
    y = 5*X + noise_var*np.random.randn(N,1) # N-by-1 X matrix as we have linear input
    
    # Define model
#    m = BayesianLinearRegression(X, y, alpha, beta)
    blrf = BayesianLinearRegressionFeatures()
    m = blrf.BuildIdentityFeaturedModel(X, y, M, (alpha, beta))
    
    # Fit MLE and MAP estimates for w
    w_MLE = m.fit_MLE()
    w_MAP, Lambda_inv = m.fit_MAP()
    
    # Predict at a set of test points
    X_star = np.linspace(0,1,200)[:, None] # our x_star points will depend on the phi(x) function
    X_star = blrf.FormatX_star(X_star, M)
    y_pred_MLE = np.matmul(X_star, w_MLE)
    y_pred_MAP = np.matmul(X_star, w_MAP)
    
    # Draw sampes from the predictive posterior
    num_samples = 500
    mean_star, var_star = m.predictive_distribution(X_star)
    samples = np.random.multivariate_normal(mean_star.flatten(), var_star, num_samples)
    
    # Plot
    plt.figure(1, figsize=(8,6))
    plt.subplot(1,2,1)
    plt.plot(X_star, y_pred_MLE, linewidth=3.0, label = 'MLE')
    plt.plot(X_star, y_pred_MAP, linewidth=3.0, label = 'MAP')
    for i in range(0, num_samples):
        plt.plot(X_star, samples[i,:], 'k', linewidth=0.05)
    plt.plot(X,y,'o', label = 'Data')
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
#    plt.axis('tight')


    print("X_star shape: {}, w_MLE shape: {}, w_MAP: {}".format(X_star.shape, w_MLE.shape, w_MAP.shape))
    # Plot distribution of w
    plt.subplot(1,2,2)
    x_axis = np.linspace(4, 5.5, 1000)[:,None]
    plt.plot(x_axis, norm.pdf(x_axis,w_MAP,Lambda_inv), label = 'p(w|D)')
    plt.legend()
    plt.xlabel('$w$')
    plt.ylabel('$p(w|D)$')
    plt.axis('tight')