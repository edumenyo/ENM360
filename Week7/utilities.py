#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 13:06:20 2018

@author: paris
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt

def plot_random_sample(images, labels):
        idx = np.random.randint(images.shape[0])
        plt.figure()
        plt.clf()
        img = images[idx,:].reshape([28,28])
        plt.imshow(img, cmap=plt.get_cmap('gray_r'))
        plt.title('This is a %d' % labels[idx,:].argmax())
        plt.show()

def one_hot(y):
    N, C = y.shape[0],y.max(0)+1
    Y = np.zeros((N,C))
    for i in range(0, N):
        idx = y[i]
        Y[i,idx] = 1
    return Y

def deone_hot(Y):
    N = Y.shape[0]
    y = np.zeros(N)
    for i in range(0, N):
        y[i] = Y[i,:].argmax()
    return y

    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


