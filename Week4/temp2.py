# /////////////////////////////////////////////
# /////////////////////////////////////////////
# /////////////////////////////////////////////
# N = 50, M = 17, monomial basis

from models import BayesianLinearRegression
from features import BayesianLinearRegressionFeatures
import numpy as np
import math
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.stats import norm

N = 200 # N is the number of training points.
M = 5   # M is the number of parameters (a.k.a the length of w)
alpha = 5.0
beta = 0.1

# Create random input and output data // will probably need to use a function for normal dist here
X = lhs(1, N)*3
epsilon = norm.pdf(X, scale=0.5)
print("X shape: {}, epsilon shape: {}".format(X.shape, epsilon.shape))
y = np.exp(X) * np.sin(2 * np.pi * X) + epsilon

# Define models
blrf = BayesianLinearRegressionFeatures()
m = blrf.BuildLegendreFeaturedModel(X, y, M, (alpha, beta))

# Fit MLE and MAP estimates for w
w_MLE = m.fit_MLE()
w_MAP, Lambda_inv = m.fit_MAP()

# Predict at a set of test points
X_star = np.linspace(0,3,N)[:, None] # our x_star points will depend on the phi(x) function
Phi_star = blrf.FormatX_star(X_star, M)


print("X_star shape: {}, w_MLE shape: {}".format(X_star.shape, w_MLE.shape))
y_pred_MLE = np.matmul(Phi_star, w_MLE)
y_pred_MAP = np.matmul(Phi_star, w_MAP)

# Draw sampes from the predictive posterior
num_samples = 500
mean_star, var_star = m.predictive_distribution(Phi_star)
samples = np.random.multivariate_normal(mean_star.flatten(), var_star, num_samples)

# Plot
plt.figure(1, figsize=(10,6))
# plt.subplot(1,2,1)
for i in range(0, num_samples):
    plt.plot(X_star, samples[i,:], 'k', linewidth=0.05)
plt.plot(X_star, y_pred_MLE, linewidth=1.0, label = 'MLE')
plt.plot(X_star, y_pred_MAP, linewidth=1.0, label = 'MAP')
plt.plot(X,y,'o', label = 'Data')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$y$')
#    plt.axis('tight')

plt.subplot(2,2,1)

# Plot distribution of w
# # plt.subplot(1,2,2)
# plt.figure(2, figsize=(10,6))
# x_axis = np.linspace(-1.5, 15, N)[:,None]

# for i in range(w_MLE[:,].size):
#     x_axis = np.linspace(w_MLE[i].min(), w_MLE[i].max(), w_MLE[:,].size)[:,None]
#     plt.plot(x_axis, w_MLE[i], label='$\phi_{}(x)$'.format(i))

# # for i in range(w_MLE[:,].size):
# #     plt.plot(x_axis, norm.pdf(x_axis, w_MAP.T, Lambda_inv[i][i])[:,i], label = 'p_{}(w|D)'.format(i))
# plt.legend()
# plt.xlabel('$w$')
# plt.ylabel('$p(w|D)$')
# #     plt.axis('tight')