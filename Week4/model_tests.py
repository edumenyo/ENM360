# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 23:05:40 2019

@author: pensi
"""

from models import BayesianLinearRegression
from features import BayesianLinearRegressionFeatures
import numpy as np
import math
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.stats import norm

def testFourier(num_samples):
    raise NotImplementedError()
    
    # Define models
    blrf = BayesianLinearRegressionFeatures()
    m = blrf.BuildIdentityFeaturedModel(X, y, M, (alpha, beta))
    
    # Fit MLE and MAP estimates for w
    w_MLE = m.fit_MLE()
    w_MAP, Lambda_inv = m.fit_MAP()
    
    # Predict at a set of test points
    X_star = np.linspace(0,3,N)[:, None] # our x_star points will depend on the phi(x) function
    Phi_star = blrf.FormatX_star(X_star, M)
    
    
    print("X_star shape: {}, w_MLE shape: {}".format(X_star.shape, 
          w_MLE.shape))
    y_pred_MLE = np.matmul(X_star, w_MLE)
    y_pred_MAP = np.matmul(X_star, w_MAP)
    
    # Draw sampes from the predictive posterior
#    num_samples = 1
    mean_star, var_star = m.predictive_distribution(X_star)
    samples = np.random.multivariate_normal(mean_star.flatten(), 
                                            var_star, 
                                            num_samples)
        
    return (X_star, Phi_star, (y_pred_MLE, y_pred_MAP), samples)
    
def testLegendre(num_samples):
    raise NotImplementedError()
    
    # Define models
    blrf = BayesianLinearRegressionFeatures()
    m = blrf.BuildIdentityFeaturedModel(X, y, M, (alpha, beta))
    
    # Fit MLE and MAP estimates for w
    w_MLE = m.fit_MLE()
    w_MAP, Lambda_inv = m.fit_MAP()
    
    # Predict at a set of test points
    X_star = np.linspace(0,3,N)[:, None] # our x_star points will depend on the phi(x) function
    Phi_star = blrf.FormatX_star(X_star, M)
    
    
    print("X_star shape: {}, w_MLE shape: {}".format(X_star.shape, 
          w_MLE.shape))
    y_pred_MLE = np.matmul(X_star, w_MLE)
    y_pred_MAP = np.matmul(X_star, w_MAP)
    
    # Draw sampes from the predictive posterior
#    num_samples = 1
    mean_star, var_star = m.predictive_distribution(X_star)
    samples = np.random.multivariate_normal(mean_star.flatten(),
                                            var_star, num_samples)

    
    return (X_star, Phi_star, (y_pred_MLE, y_pred_MAP), samples)
    
    

if __name__ == "__main__":
    
    N = 500 # N is the number of training points.
    M = 4   # M is the number of parameters (a.k.a the length of w)
    alpha = 5.0
    beta = 0.1
    noise_var = 0.3
    
    X = lhs(1, N)*3
    epsilon = norm.pdf(X, scale=math.sqrt(0.5))
    y = np.exp(X) * np.sin(2 * np.pi * X) + epsilon
    
    (X_star, Phi_star, (y_pred_MLE, y_pred_MAP), samples) = testFourier(500);
    
    # Plot
    plt.figure(1, figsize=(10,6))
    plt.subplot(1,2,1)
    plt.plot(X_star, y_pred_MLE, linewidth=0.75, label = 'MLE')
    plt.plot(X_star, y_pred_MAP, linewidth=0.75, label = 'MAP')
    for i in range(0, samples[0,:].size):
        plt.plot(X_star, samples[i,:], 'k', linewidth=0.33)
    plt.plot(X,y,'o', label = 'Data')
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    #    plt.axis('tight')
    
    # /////////////////////////////////////////////
    
    (X_star, Phi_star, (y_pred_MLE, y_pred_MAP), samples) = testLegendre(500);

    # Plot
    plt.figure(2, figsize=(10,6))
    plt.subplot(1,2,1)
    plt.plot(X_star, y_pred_MLE, linewidth=0.75, label = 'MLE')
    plt.plot(X_star, y_pred_MAP, linewidth=0.75, label = 'MAP')
    for i in range(0, samples[0,:].size):
        plt.plot(X_star, samples[i,:], 'k', linewidth=0.33)
    plt.plot(X,y,'o', label = 'Data')
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    #    plt.axis('tight')

### Pre-thoughts: Of course one of the dangers here is overfitting. Given that the Fourier basis is composed
    # of sinusoids, I would expect it to do the best with approximation.

# Experiment 1:

# Experiment 2:

# Experiment 3:

# Experiment 4: (?)

# =======================
# Plotting features according to each basis for N=500, M=4

#N = 500 # N is the number of training points.
#M = 4   # M is the number of parameters (a.k.a the length of w)
#alpha = 5.0
#beta = 0.1
#noise_var = 0.3
#
#X = lhs(1, N)*3
#epsilon = norm.pdf(X, scale=math.sqrt(0.5))
#y = np.exp(X) * np.sin(2 * np.pi * X) + epsilon
#
## Define models
#blrf = BayesianLinearRegressionFeatures()
#m = blrf.BuildIdentityFeaturedModel(X, y, M, (alpha, beta))
##     m = BayesianLinearRegression(X, y, alpha, beta)
#
## Fit MLE and MAP estimates for w
#w_MLE = m.fit_MLE()
#w_MAP, Lambda_inv = m.fit_MAP()
#
## Predict at a set of test points
#X_star = np.linspace(0,3,N)[:, None] # our x_star points will depend on the phi(x) function
#Phi_star = blrf.FormatX_star(X_star, M)
#
#
#print("X_star shape: {}, w_MLE shape: {}".format(X_star.shape, w_MLE.shape))
#y_pred_MLE = np.matmul(X_star, w_MLE)
#y_pred_MAP = np.matmul(X_star, w_MAP)
#
## Draw sampes from the predictive posterior
#num_samples = 1
#mean_star, var_star = m.predictive_distribution(X_star)
#samples = np.random.multivariate_normal(mean_star.flatten(), var_star, num_samples)