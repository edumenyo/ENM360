# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:14:42 2019

@author: Abhinav Ramkumar
"""

import autograd.numpy as np
import torch
from torch.autograd import Variable
from autograd import grad
from optimizers import Adam
import timeit

class NeuralNetwork:
    def __init__(self, X, Y, layers):
        super(NeuralNetwork, self).__init__()
        
        # normalize the input function data
        self.Xmean, self.Xstd = X.mean(0), X.std(0)
        self.Ymean, self.Ystd = Y.mean(0), Y.std(0)
        X[:, 0] = (X[:, 0] - self.Xmean[0])/self.Xstd[0]
        X[:, 1] = (X[:, 1] - self.Xmean[1])/self.Xstd[1]
        Y = (Y - self.Ymean) / self.Ystd
        
        # define class parameters
        self.X = X
        self.X.requires_grad = False
        self.Y = Y
        self.Y.requires_grad = False
        self.layers = layers
        
        # Define and initialize neural network
        self.initialize_NN(self.layers)
        
        # Total number of parameters
        self.num_params = self.weights + self.bias
        
        # Define optimizer
        self.optimizer = torch.optim.Adam(self.num_params, lr = 1e-3)
        
    def initialize_NN(self, Q):
        num_layers = len(Q)
        self.weights = []
        self.bias = []
        for layer in range(0,num_layers-1):
            w = -np.sqrt(6.0/(Q[layer]+Q[layer+1])) + 2.0*np.sqrt(6.0/(Q[layer]+Q[layer+1]))*np.random.rand(Q[layer],Q[layer+1])
            b = np.zeros((1,Q[layer+1]))
            W = torch.from_numpy(w).type(torch.FloatTensor)
            W.requires_grad = True
            B = torch.from_numpy(b).type(torch.FloatTensor)
            B.requires_grad = True
            self.weights.append(W)
            self.bias.append(B)
            
    def forward_pass(self, X, Q, weights, bias):
        H = X
        for i in range(0,len(Q)-2):
            self.z = H.mm(self.weights[i]) + self.bias[i]
            H = torch.tanh(self.z) # activation function
        self.z = H.mm(self.weights[-1]) + self.bias[-1]
        return self.z

    # Evaluates the mean square error loss
    def loss(self, weights, bias):
        X = self.X_batch
        X.requires_grad = False
        Y = self.Y_batch      
        Y.requires_grad = False                    
        mu = self.forward_pass(X, self.layers, self.weights, self.bias)                
        return torch.mean((Y-mu)**2)
    
    
    # Fetches a mini-batch of data
    def fetch_minibatch(self,X, Y, N_batch):
        N = X.shape[0]
        idx = np.random.choice(N, N_batch, replace=False)
        X_batch = X[idx,:]
        Y_batch = Y[idx]        
        return X_batch, Y_batch
    
    
    # Trains the model by minimizing the MSE loss
    def train(self, nIter, batch_size):   

        start_time = timeit.default_timer()            
        for it in range(nIter):
            # Fetch minibatch
            self.X_batch, self.Y_batch = self.fetch_minibatch(self.X, self.Y, batch_size)
            self.X_batch.requires_grad = False
            self.Y_batch.requires_grad = False
            self.optimizer.zero_grad()
            
            # Evaluate loss using current parameters
            loss = self.loss(self.weights,self.bias)
            loss.backward()
            self.optimizer.step()
            
            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss, elapsed))
                start_time = timeit.default_timer()
            
      
    # Evaluates predictions at test points              
    def predict(self, X_star): 
        # Normalize inputs
        X_star[0] = (X_star[0] - self.Xmean[0])/self.Xstd[0]
        X_star[1] = (X_star[1] - self.Xmean[1])/self.Xstd[1]            
        y_star = self.forward_pass(X_star, self.layers, self.weights, self.bias)
        # De-normalize outputs
        y_star = y_star*self.Ystd + self.Ymean            
        return y_star