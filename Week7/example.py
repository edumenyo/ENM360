#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:20:02 2018

@author: Paris
"""

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

#from models_numpy import NeuralNetwork
from models_pytorch import NeuralNetwork
#from models_tf import NeuralNetwork


if __name__ == "__main__": 
    
    N = 200
    X_dim = 1
    Y_dim = 1
    layers = [X_dim,256,512,256,512,256,Y_dim]
    noise = 0.0
    
    # Generate Training Data   
    def f(x):
        return np.sin(8.0*np.pi*x)
    
    # Specify input domain bounds
    lb = 0.0*np.ones((1,X_dim))
    ub = 1.0*np.ones((1,X_dim)) 
    
    # Generate data
#    X = lb + (ub-lb)*lhs(X_dim, N)
    X = np.linspace(lb[0], ub[0], N)[:,None]
    Y = f(X) + noise*np.random.randn(N,Y_dim)
    
    # Generate Test Data
    N_star = 1000
    X_star = lb + (ub-lb)*np.linspace(0,1,N_star)[:,None]
    Y_star = f(X_star)
            
    # Create model
    model = NeuralNetwork(X, Y, layers)
        
    # Training
    model.train(nIter = 40000, batch_size = N)
    
    # Prediction
    Y_pred = model.predict(X_star)
    
    # Plotting
    plt.figure(1)
    plt.plot(X_star, Y_star, 'b-', linewidth=2)
    plt.plot(X_star, Y_pred, 'r--', linewidth=2)
    plt.scatter(X, Y, alpha = 0.8)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(['$f(x)$', 'prediction', '%d training data' % N], loc='lower left')
