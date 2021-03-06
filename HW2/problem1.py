import math
import numpy as np
from collections import Counter
#-------------------------------------------------------------------------
'''
    Problem 1: k nearest neighbor 
    In this problem, you will implement a classification method using k nearest neighbors. 
    The main goal of this problem is to get familiar with the basic settings of classification problems. 
    KNN is a simple method for classification problems.
    You could test the correctness of your code by typing `nosetests test1.py` in the terminal.
'''

#--------------------------
def compute_distance(Xtrain, Xtest):
    '''
        compute the Euclidean distance between instances in a test set and a training set 
        Input:
            Xtrain: the feature matrix of the training dataset, a float python matrix of shape (n_train by p). Here n_train is the number of data instance in the training set, p is the number of features/dimensions.
            Xtest: the feature matrix of the test dataset, a float python matrix of shape (n_test by p). Here n_test is the number of data instance in the test set, p is the number of features/dimensions.
        Output:
            D: the distance between instances in Xtest and Xtrain, a float python matrix of shape (ntest, ntrain), the (i,j)-th element of D represents the Euclidean distance between the i-th instance in Xtest and j-th instance in Xtrain.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    n_test=len(Xtest)
    n_train=len(Xtrain)
    p=Xtest.shape[1]
    D=np.zeros((n_test, n_train))
    i=0
    for test_row in Xtest:
        j=0
        for train_row in Xtrain:
            diff_sum=0
            for k in range(p):
                diff_sum+=(test_row[k]-train_row[k])**2
            D[i][j]=np.sqrt(diff_sum)
            j+=1
        i+=1    
    #########################################
    return D 



#--------------------------
def k_nearest_neighbor(Xtrain, Ytrain, Xtest, K = 3):
    '''
        compute the labels of test data using the K nearest neighbor classifier.
        Input:
            Xtrain: the feature matrix of the training dataset, a float numpy matrix of shape (n_train by p). Here n_train is the number of data instance in the training set, p is the number of features/dimensions.
            Ytrain: the label vector of the training dataset, an integer python list of length n_train. Each element in the list represents the label of the training instance. The values can be 0, ..., or num_class-1. num_class is the number of classes in the dataset.
            Xtest: the feature matrix of the test dataset, a float python matrix of shape (n_test by p). Here n_test is the number of data instance in the test set, p is the number of features/dimensions.
            K: the number of neighbors to consider for classification.
        Output:
            Ytest: the predicted labels of test data, an integer numpy vector of length ntest.
        Note: you cannot use any existing package for KNN classifier.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    Ytest=[]
    distances=compute_distance(Xtrain, Xtest)
    for row in distances:
        k_indices=np.argsort(row)[::-1][-K:]
        neighbors_classes={}
        for i in k_indices:
            y_class=Ytrain[i]
            if y_class in neighbors_classes:
                neighbors_classes[y_class]+=1
            else:
                neighbors_classes[y_class]=1
        y_hat=max(neighbors_classes, key=neighbors_classes.get)
        Ytest.append(y_hat)
    
    #########################################
    return np.asarray(Ytest)

