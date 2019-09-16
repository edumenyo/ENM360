# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 22:32:53 2019

2.) Write a Python class for Lagrange interpolation and use it to interpolate 
    the training data provided here (HW2_training_data.txt). Plot the training 
    data, the resulting Lagrange interpolant, and provide a brief comment on 
    the prediction accuracy tested against a set of validation data provided 
    here (HW2_validation_data.txt).

@author: edumenyo
"""

class LagrangeInterpolationTool:
    def __init__():
        raise NotImplementedError()
    
    def train():
        raise NotImplementedError()
        
    def interpolate():
        raise NotImplementedError()
    
if __name__ == "__main__":
    # should take in order number (k), 
    lit = LagrangeInterpolationTool()
    
    # can take in a file, or use the default given in the assignment
    lit.train() 
    # should take in a function to interpolate (f), start of interval , 
    #   end of the interval, the number of nodes (n), 
    #   which Runge correction to use in 
    lit.interpolate() 