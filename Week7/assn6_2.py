#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 12:20:04 2018

@author: paris
"""

import numpy as np
import pickle, gzip
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from utilities import one_hot, deone_hot, plot_random_sample, plot_confusion_matrix

from models_tf import NeuralNetClassification

if __name__ == "__main__": 
    
    # Load the dataset
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    f.close()
    
    # Training data
    N_train = train_set[0].shape[0]
    train_images = train_set[0]
    train_labels = one_hot(train_set[1])
    
    # Test data
    N_test = test_set[0].shape[0]
    test_images = test_set[0]
    test_labels = one_hot(test_set[1])
        
    # Check a few samples to make sure the data was loaded correctly
    plot_random_sample(train_images, train_labels)
    
    # Model
    X_dim = train_images.shape[1]
    Y_dim = train_labels.shape[1]
    layers = [X_dim, 128, 128, Y_dim]
    model  = NeuralNetClassification(train_images, train_labels, 
                                     test_images, test_labels, 
                                     layers)
    
    # Training
    model.train(nIter = 10000, batch_size = 128)
    
    # Predictions
    pred_labels = model.predict(test_images)
    
    # Confusion matrix
    M = confusion_matrix(deone_hot(test_labels), deone_hot(pred_labels))
    
    # False positives, False negatives, True positives, True negatives
    FP = M.sum(axis=0) - np.diag(M)  
    FN = M.sum(axis=1) - np.diag(M)
    TP = np.diag(M)
    TN = M.sum() - (FP + FN + TP)
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    print('Classification accuracy: %.3f%%' % (ACC.mean(0)))
    
    # Plot loss
    plt.figure()
    plt.plot(model.training_loss, label = 'Training loss')
    plt.plot(model.testing_loss, label = 'Testing loss')
    plt.legend()
    plt.xlabel('Iter #')
    plt.ylabel('Loss')
    
    
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(M, classes=np.arange(Y_dim), normalize=False)

    