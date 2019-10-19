#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:20:02 2018

@author: Paris
"""

import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from pyDOE import lhs

#from models_numpy import NeuralNetwork
from models_pytorch import NeuralNetwork
# from models_tf import NeuralNetwork


if __name__ == "__main__":
    N = 100
    X_dim = 2
    Y_dim = 1
    layers = [X_dim, 50, 50, Y_dim]
    noise = 0.0

    # Generate Training Data
    def f(x, y):
        return np.sin(np.pi*x) * np.sin(np.pi*y)

    # Specify input domain bounds
    lb = 49
    ub = 53

    # Generate data
    x = lb + (ub-lb)*lhs(X_dim, N)
    XY = torch.from_numpy(x).type(torch.FloatTensor)
#    X = np.linspace(lb[0], ub[0], N)[:, None]
    func = torch.from_numpy(f(x[:, 0], x[:, 1])).type(torch.FloatTensor)

    # Generate Test Data
    N_star = 1000 * X_dim
    XY_star = lb + (ub-lb)*np.linspace(0, 1, N_star).reshape(-1, X_dim)
    XY_star = torch.from_numpy(XY_star).type(torch.FloatTensor)

    # Create model
    model = NeuralNetwork(XY, func, layers)

    # Training
    model.train(nIter=100, batch_size=N)

    # Prediction
    func_pred = model.predict(XY_star)
    
    print("shapes\nXY: {}, func: {}, XY_star: {}, func_pred: {}".format(
            XY.shape, func.shape, XY_star.shape, func_pred.shape))

    # Plotting
    plt.figure(1)
    plt.plot(XY_star, func, 'b-', linewidth=2)
    plt.plot(XY_star, func_pred, 'r--', linewidth=2)
    plt.scatter(XY[:, 0], XY[:, 1], alpha=0.8)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(['$f(x)$', 'prediction',
                '%d training data' % N], loc='lower left')
