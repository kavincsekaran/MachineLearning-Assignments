import math
import numpy as np
from problem1 import Tree
#-------------------------------------------------------------------------
'''
    Problem 2: Decision Tree (with continous attributes)
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
'''

#--------------------------
class Node:
    '''
        Decision Tree Node (with continous attributes)
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar 
            th: the threshold on the attribute, a float scalar.
            C1: the child node for values smaller than threshold
            C2: the child node for values larger than threshold
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X=None,Y=None, i=None,th=None,C1=None, C2=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.th = th 
        self.C1= C1
        self.C2= C2
        self.isleaf = isleaf
        self.p = p


#-----------------------------------------------
class DT(Tree):
    '''
        Decision Tree (with contineous attributes)
        Hint: DT is a subclass of Tree class in problem1. So you can reuse and overwrite the code in problem 1.
    '''

    #--------------------------
    @staticmethod
    def cutting_points(X,Y):
        '''
            Find all possible cutting points in the continous attribute of X. 
            (1) sort unique attribute values in X, like, x1, x2, ..., xn
            (2) consider splitting points of form (xi + x(i+1))/2 
            (3) only consider splitting between instances of different classes
            (4) if there is no candidate cutting point above, use -inf as a default cutting point.
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                cp: the list of  potential cutting points, a float numpy vector. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        
        
        cp=[]
        sorted_inputs=zip(*sorted(set(zip(X, Y))))
        sorted_x=list(sorted_inputs[0])
        sorted_y=list(sorted_inputs[1])
        unique_inputs={}
        multiple_class=[]
        for i in range(len(sorted_x)):
            if(sorted_x[i] in unique_inputs.keys()):
                multiple_class.append(True)
            else:
                multiple_class.append(False)
            unique_inputs[sorted_x[i]]=sorted_y[i]
        sorted_x=sorted(unique_inputs.keys())
        sorted_y=[unique_inputs.get(x) for x in sorted_x]
        
        for i in range(len(sorted_y)-1):
            if(sorted_y[i]!=sorted_y[i+1] or multiple_class[i+1]):
                cp.append((sorted_x[i]+sorted_x[i+1])/2)
        if(not cp):
            cp.append(-float("inf"))
        cp=np.array(cp)
        
        
        #########################################
        return cp
    
    #--------------------------
    @staticmethod
    def best_threshold(X,Y):
        '''
            Find the best threshold among all possible cutting points in the continous attribute of X. 
            Input:
                X: a list of values, a numpy array of int/float values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                th: the best threhold, a float scalar. 
                g: the information gain by using the best threhold, a float scalar. 
            Hint: you can reuse your code in problem 1.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        ig=[]
        cp=DT.cutting_points(X, Y)
        for x in cp:
            if x == -float("inf"):
                ig.append(-1)
            else:
                ig.append(Tree.information_gain(Y,np.less(X,x)))
        i = np.argmax(ig)
        th=cp[i]
        g=ig[i]

        #########################################
        return th,g 
    
    
    #--------------------------
    def best_attribute(self,X,Y):
        '''
            Find the best attribute to split the node. The attributes have continous values (int/float).
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
                th: the threshold of the attribute to split, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
    
   
        
        ig=[]
        ths=[]
        for x in X:
            #print(x)
            #print(Y)
            th, g =DT.best_threshold(x,Y)
            #print(g)
            #print(th)
            ig.append(g)
            ths.append(th)
        i = np.argmax(ig)
        th=ths[i]


 
        #########################################
        return i, th
    


        
    #--------------------------
    @staticmethod
    def split(X,Y,i,th):
        '''
            Split the node based upon the i-th attribute and its threshold.
            (1) split the matrix X based upon the values in i-th attribute and threshold
            (2) split the labels Y 
            (3) build children nodes by assigning a submatrix of X and Y to each node
    
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C1: the child node for values smaller than threshold
                C2: the child node for values larger than (or equal to) threshold
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        i_col=X[i,:]
        c1_indices=np.less(i_col,th)
        c2_indices=np.greater_equal(i_col,th)
        C1=Node(X[:,c1_indices],Y[c1_indices])
        C2=Node(X[:,c2_indices],Y[c2_indices])
        '''
        print(X)
        print(i)
        print(th)
        print(C1.X)
        print(C2.X)
        '''
        #########################################
        return C1, C2
    
    
    
    #--------------------------
    def build_tree(self, t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape n by p.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C1: the child node for values smaller than threshold
                t.C2: the child node for values larger than (or equal to) threshold
        '''
        #########################################
        ## INSERT YOUR CODE HERE
    
        # if Condition 1 or 2 holds, stop recursion 
        t.p=Tree.most_common(t.Y)
        if(Tree.stop1(t.Y) or Tree.stop2(t.X)):
            t.isleaf=True
        else:
            #print(t.X)
            #print(t.Y)
            t.i, t.th=self.best_attribute(t.X, t.Y)
            #print(t.th)
            t.C1, t.C2=DT.split(t.X, t.Y, t.i, t.th)
            self.build_tree(t.C2)
            self.build_tree(t.C1)
            
 
        # find the best attribute to split



        # recursively build subtree on each child node
   

 
        #########################################
    
    #--------------------------
    @staticmethod
    def inference(t,x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively. 
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
    
        if((t.C1==None and t.C2==None) or t.isleaf):
            y=t.p
        
        else:
            if(x[t.i]<t.th):
                y=DT.inference(t.C1, x)
            else:
                y=DT.inference(t.C2, x)
        

        #########################################
        return y
    
    
    #--------------------------
    @staticmethod
    def predict(t,X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset. 
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
    

        Y=[]
        for x in X.T:
            Y.append(DT.inference(t, x))
        
        #########################################
        return np.array(Y)
    
    
    
    #--------------------------
    def train(self, X, Y):
        '''
            Given a training set, train a decision tree. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t=Node(X, Y)
        self.build_tree(t)
   
        #########################################
        return t


    #--------------------------
    @staticmethod
    def load_dataset(filename='data2.csv'):
        '''
            Load dataset 2 from the CSV file: data2.csv. 
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        X = (np.genfromtxt(filename, skip_header=1, delimiter=",", usecols=(1,2,3,4,5,6,7))).T
        Y = np.genfromtxt(filename, skip_header=1, delimiter=",",usecols=(0),dtype=str)


 
        #########################################
        return X,Y




