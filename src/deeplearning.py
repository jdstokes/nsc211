#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:16:26 2017

@author: jdstokes
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error

def mse1(targets,predictions):
    m = len(targets)
    return np.sum((targets - predictions) ** 2)/m
    
def mse2(targets,predictions):
    return ((targets - predictions) ** 2).mean()    
    
#T = np.array([1,2,3])
#P = np.array([4,5,6])
#
#print(mse2(T,P))

#calculate the determinant
#M = np.array([[3,8],[4, 6]])
#print(M)
#print(np.linalg.det(M))


##linear regression example
#diabetes = datasets.load_diabetes()
#
## Use only one feature
#diabetes_X = diabetes.data[:, np.newaxis, 2]
#
## Split the data into training/testing sets
#diabetes_X_train = diabetes_X[:-20]
#diabetes_X_test = diabetes_X[-20:]
#
## Split the targets into training/testing sets
#diabetes_y_train = diabetes.target[:-20]
#diabetes_y_test = diabetes.target[-20:]
#
#
#num_samples = len(diabetes_X_train)
#X = np.concatenate((np.ones(diabetes_X_train.shape),diabetes_X_train),1)
#Betas = np.dot(np.transpose(X), X)


#numerical stability
#a = 1000000000
#for i in xrange(1000000):
#    a = a + 0.000001 #1e-6
    
   #should be 1 but it's ~.95 
#print(a - 1000000000)

#how do we get around this??





#cross entropy: used to measure the distance between two vectors
# cross entropy is not symmetric
# there's a nost log

def cross_entropy(S,L):
   return -1* sum(L*np.log(S))



S = np.array([0.7,0.2,0.1])
L = np.array([1.0,0.0,0.0])

print(cross_entropy(S,L))





























