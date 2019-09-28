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
        
        if self.basisType == "identity":
            return X_star
        elif self.basisType == "monomial":
            newMat = np.zeros((X_star.size, num_features+1))
            for i in range(X_star.size):
                for j in range(num_features+1):
                    newMat[i][j] = math.pow(X_star[i], j)
#                     print("\trow size: {}", features[i].size)

            return newMat
      
    def BuildMonomialFeaturedModel(self, X_train, Y_train, 
                                   num_features, consts=None):
#        raise NotImplementedError()
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
        features = np.array([basisFunc(obs) for obs in X_train])
        self.basisType = "legendre"
        return BayesianLinearRegression(features, Y_train, consts[0], consts[1])  

if __name__ == "__main__": 
    
    N = 500 # N is the number of training points.
    M = 8   # M is the number of parameters (a.k.a the length of w)
    alpha = 5.0
    beta = 0.1
    noise_var = 0.3
    
    # Create random input and output data // will probably need to use a function for normal dist here
    X = lhs(1, N)
#     y = 5*X + noise_var*np.random.randn(N,1) # N-by-1 X matrix as we have linear input
    y = np.exp(X) * np.sin(2 * np.pi * X)
    
    # Define models
    blrf = BayesianLinearRegressionFeatures()
    m = blrf.BuildMonomialFeaturedModel(X, y, M, (alpha, beta))
#     m = BayesianLinearRegression(X, y, alpha, beta)
    
    # Fit MLE and MAP estimates for w
    w_MLE = m.fit_MLE()
    w_MAP, Lambda_inv = m.fit_MAP()
    
    # Predict at a set of test points
    X_star = np.linspace(0,3,N)[:, None] # our x_star points will depend on the phi(x) function
    X_star = blrf.FormatX_star(X_star, M)
    
    
    print("X_star shape: {}, w_MLE shape: {}".format(X_star.shape, w_MLE.shape))
    y_pred_MLE = np.matmul(X_star, w_MLE)
    y_pred_MAP = np.matmul(X_star, w_MAP)
    
    # Draw sampes from the predictive posterior
    num_samples = 500
    mean_star, var_star = m.predictive_distribution(X_star)
    print("var_star: {}".format(var_star))
    samples = np.random.multivariate_normal(mean_star.flatten(), var_star, num_samples)
    
    # Plot
    plt.figure(1, figsize=(10,6))
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


    # Plot distribution of w
    plt.subplot(1,2,2)
    x_axis = np.linspace(-1.5, 10, N)[:,None]
    for i in range(M+1):
        plt.plot(x_axis, norm.pdf(x_axis, w_MAP.T, Lambda_inv[i][i])[:,i], label = 'p(w|D)')
    plt.legend()
    plt.xlabel('$w$')
    plt.ylabel('$p(w|D)$')
#     plt.axis('tight')