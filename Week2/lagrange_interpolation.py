#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 17:25:35 2018

@author: paris
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    # Create a function computing the Lagrange 
    # interpolating polynomial
    def LagrangeInpterolant(x_star, x, y):
        return
            
    # Load training data
    training_data = np.loadtxt('HW2_training_data.txt')
    x_t = training_data[:,0:1]
    y_t = training_data[:,1:2]
    
    # Load validation data
    validation_data = np.loadtxt('HW2_validation_data.txt')
    x_v = validation_data[:,0:1]
    y_v = validation_data[:,1:2]
    
    # Create a grid to plot the predicted function
    x_star = np.linspace(x_v.min(0), x_v.max(0), 100)
    
    # Construct the Lagrange interpolant
    y_pred = LagrangeInpterolant(x_star, x_t, y_t)

    # Plot the results
    plt.figure(1)
    plt.plot(x_v, y_v,'bo', markerfacecolor = 'None', label = "Validation data")
    plt.plot(x_t, y_t,'rx', label = "Training data")
    plt.plot(x_star, y_pred, 'k-', label = "Lagrange interpolant")
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.legend()
    
       
    
