from problem1 import SoftmaxRegression as sr
import torch as th
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.optim import SGD
import numpy as np
import math

#-------------------------------------------------------------------------
'''
    Problem 2: Convolutional Neural Network 
    In this problem, you will implement a convolutional neural network with a convolution layer and a max pooling layer.
    The goal of this problem is to learn the details of convolutional neural network. 
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
    Note: please do NOT use th.nn.functional.conv2d or th.nn.Conv2D, implement your own version of 2d convolution using only basic tensor operations.
'''

#--------------------------
def conv2d(x,W,b):
    '''
        Compute the 2D convolution with one filter on one image, (assuming stride=1).
        Input:
            x:  one training instance, a float torch Tensor of shape l by h by w. 
                h and w are height and width of an image. l is the number channels of the image, for example RGB color images have 3 channels.
            W: the weight matrix of a convolutional filter, a torch Variable of shape l by s by s. 
            b: the bias vector of the convolutional filter, a torch scalar Variable. 
        Output:
            z: the linear logit tensor after convolution, a float torch Variable of shape (h-s+1) by (w-s+1)
        Note: please do NOT use th.nn.functional.conv2d, implement your own version of 2d convolution,using basic tensor operation, such as dot().
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    W.requires_grad=True
    b.requires_grad=True
    
    l, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    s=int(W.shape[1])

    z_height = h-s+1
    z_width = w-s+1
    
    rows = np.tile(np.repeat(np.arange(s), s), l).reshape(-1, 1) + np.repeat(np.arange(z_height), z_width).reshape(1, -1)
    
    columns = np.tile(np.arange(s), s * l).reshape(-1, 1) + np.tile(np.arange(z_width), z_height).reshape(1, -1)
    
    channels = np.repeat(np.arange(l), s * s).reshape(-1, 1)
    
    conv = x[channels, rows, columns]
    z=th.mm(W.view(1, -1), conv)+b
    z=z.view(z_height,z_width)
   
    #########################################
    return z 


#--------------------------
def Conv2D(x,W,b):
    '''
        Compute the 2D convolution with multiple filters on a batch of images, (assuming stride=1).
        Input:
            x:  a batch of training instances, a float torch Tensor of shape (n by l by h by w). n is the number instances in a batch.
                h and w are height and width of an image. l is the number channels of the image, for example RGB color images have 3 channels.
            W: the weight matrix of a convolutional filter, a torch Variable of shape (n_filters by l by s by s). 
            b: the bias vector of the convolutional filter, a torch vector Variable of length n_filters. 
        Output:
            z: the linear logit tensor after convolution, a float torch Variable of shape (n by n_filters by (h-s+1) by (w-s+1) )
        Note: please do NOT use th.nn.functional.conv2d, implement your own version of 2d convolution,using basic tensor operation, such as dot().
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    #print "size of W", W.size()
    #print "val of W", W
    n, l, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
    n_filter=int(W.shape[0])
    #print "n_filter", n_filter, W.shape[-1]
    s=int(W.shape[-1])
    #print W.view(n_filter, -1)

    z_height = h-s+1
    z_width = w-s+1

    rows = np.tile(np.repeat(np.arange(s), s), l).reshape(-1, 1) + np.repeat(np.arange(z_height), z_width).reshape(1, -1)

    columns = np.tile(np.arange(s), s * l).reshape(-1, 1) + np.tile(np.arange(z_width), z_height).reshape(1, -1)

    channels = np.repeat(np.arange(l), s * s).reshape(-1, 1)

    conv = x[:, channels, rows, columns]
    
    zs=[]
    for image in conv:
        z_n=th.mm(W.view(n_filter, -1), image)
        z_n=z_n+th.t(b.expand(int(z_n.shape[1]), n_filter))
        z_n=z_n.view(n_filter, z_height,z_width)
        zs.append(z_n)
    z=th.cat(zs).view(n, n_filter, z_height,z_width)
    
    #########################################
    return z 


#--------------------------
def ReLU(z):
    '''
        Compute ReLU activation. 
        Input:
            z: the linear logit tensor after convolution, a float torch Variable of shape (n by n_filters by h by w )
                h and w are the height and width of the image after convolution. 
        Output:
            a: the nonlinear activation tensor, a float torch Variable of shape (n by n_filters by h by w )
        Note: please do NOT use th.nn.functional.relu, implement your own version using only basic tensor operations. 
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    zeros_var=Variable(th.zeros(z.shape))
    a=th.max(zeros_var, z)
    #########################################
    return a 


#--------------------------
def avgpooling(a):
    '''
        Compute the 2D average pooling (assuming shape of the pooling window is 2 by 2).
        Input:
            a:  the feature map of one instance, a float torch Tensor of shape (n by n_filter by h by w). n is the batch size, n_filter is the number of filters in Conv2D.
                h and w are height and width after ReLU. 
        Output:
            p: the tensor after pooling, a float torch Variable of shape n by n_filter by floor(h/2) by floor(w/2).
        Note: please do NOT use torch.nn.AvgPool2d or torch.nn.functional.avg_pool2d or avg_pool1d, implement your own version using only basic tensor operations.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    
    n, n_filter, h, w = int(a.shape[0]), int(a.shape[1]), int(a.shape[2]), int(a.shape[3])
    
    #pooling_window size
    s=2
    l=1

    z_height = int(math.floor(h/2))
    z_width = int(math.floor(w/2))

    rows = np.tile(np.repeat(np.arange(s), s), l).reshape(-1, 1) + s*np.repeat(np.arange(z_height), z_width).reshape(1, -1)

    columns = np.tile(np.arange(s), s * l).reshape(-1, 1) + s*np.tile(np.arange(z_width), z_height).reshape(1, -1)

    conv = a[:, :, rows, columns]
    
    zs=[]
    for image in conv:
        print(image)
        for fil in image:
            print(fil)
            z_n=th.mean(fil, dim=0)
            zs.append(z_n)
    p=th.cat(zs).view(n, n_filter, z_height,z_width)

    #########################################
    return p 

#--------------------------
def maxpooling(a):
    '''
        Compute the 2D max pooling (assuming shape of the pooling window is 2 by 2).
        Input:
            a:  the feature map of one instance, a float torch Tensor of shape (n by n_filter by h by w). n is the batch size, n_filter is the number of filters in Conv2D.
                h and w are height and width after ReLU. 
        Output:
            p: the tensor after max pooling, a float torch Variable of shape n by n_filter by floor(h/2) by floor(w/2).
        Note: please do NOT use torch.nn.MaxPool2d or torch.nn.functional.max_pool2d or max_pool1d, implement your own version using only basic tensor operations.
        Note: if there are mulitple max values, select the one with the smallest index.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    n, n_filter, h, w = int(a.shape[0]), int(a.shape[1]), int(a.shape[2]), int(a.shape[3])
    
    #pooling_window size
    s=2
    l=1

    z_height = int(math.floor(h/2))
    z_width = int(math.floor(w/2))

    rows = np.tile(np.repeat(np.arange(s), s), l).reshape(-1, 1) + s*np.repeat(np.arange(z_height), z_width).reshape(1, -1)

    columns = np.tile(np.arange(s), s * l).reshape(-1, 1) + s*np.tile(np.arange(z_width), z_height).reshape(1, -1)

    conv = a[:, :, rows, columns]
    
    zs=[]
    for image in conv:
        for fil in image:
            z_n, _=th.max(fil, dim=0)
            zs.append(z_n)
    p=th.cat(zs).view(n, n_filter, z_height,z_width)

    #########################################
    return p 


#--------------------------
def num_flat_features(h=28, w=28, s=3, n_filters=10):
    ''' Compute the number of flat features after convolution and pooling. Here we assume the stride of convolution is 1, the size of pooling kernel is 2 by 2, no padding. 
        Inputs:
            h: the hight of the input image, an integer scalar
            w: the width of the input image, an integer scalar
            s: the size of convolutional filter, an integer scalar. For example, a 3 by 3 filter has a size 3.
            n_filters: the number of convolutional filters, an integer scalar
        Outputs:
            p: the number of features we will have on each instance after convolution, pooling, and flattening, an integer scalar.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    conv=(h-s+1)
    pooled=int(math.floor(conv/2))
    p=pooled*pooled*n_filters    

    #########################################
    return p
 

#-------------------------------------------------------
class CNN(sr):
    '''CNN is a convolutional neural network with a convolution layer (with ReLU activation), a max pooling layer and a fully connected layer.
       In the convolutional layer, we will use ReLU as the activation function. 
       After the convolutional layer, we apply a 2 by 2 max pooling layer, before feeding into the fully connected layer.
    '''
    # ----------------------------------------------
    def __init__(self, l=1, h=28, w=28, s=5, n_filters=5, c=10):
        ''' Initialize the model. Create parameters of convolutional layer and fully connected layer. 
            Inputs:
                l: the number of channels in the input image, an integer scalar
                h: the hight of the input image, an integer scalar
                w: the width of the input image, an integer scalar
                s: the size of convolutional filter, an integer scalar. For example, a 3 by 3 filter has a size 3.
                n_filters: the number of convolutional filters, an integer scalar
                c: the number of output classes, an integer scalar
            Outputs:
                self.conv_W: the weight matrix of the convolutional filters, a torch Variable of shape n_filters by l by s by s, initialized as all-zeros. 
                self.conv_b: the bias vector of the convolutional filters, a torch vector Variable of length n_filters, initialized as all-ones, to avoid vanishing gradient.
                self.W: the weight matrix parameter in fully connected layer, a torch Variable of shape (p, c), initialized as all-zeros. 
                        Hint: CNN is a subclass of SoftmaxRegression, which already has a W parameter. p is the number of flat features after pooling layer.
                self.b: the bias vector parameter, a torch Variable of shape (c), initialized as all-zeros
                self.loss_fn: the loss function object for softmax regression. 
            Note: In this problem, the parameters are initialized as either all-zeros or all-ones for testing purpose only. In real-world cases, we usually initialize them with random values.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        # compute the number of flat features
        p=num_flat_features(h, w, s, n_filters)
        
        # initialize fully connected layer 
        super(CNN, self).__init__(p, c)
        
        # the kernel matrix of convolutional layer 
        self.conv_W=Variable(th.zeros(n_filters, l, s, s), requires_grad=True)
        self.conv_b=Variable(th.ones(n_filters), requires_grad=True)


        #########################################


    # ----------------------------------------------
    def forward(self, x):
        '''
           Given a batch of training instances, compute the linear logits of the outputs. 
            Input:
                x:  a batch of training instance, a float torch Tensor of shape n by l by h by w. Here n is the batch size. l is the number of channels. h and w are height and width of an image. 
            Output:
                z: the logit values of the batch of training instances after the fully connected layer, a float matrix of shape n by c. Here c is the number of classes
        '''
        #########################################
        ## INSERT YOUR CODE HERE
    
        # convolutional layer
        z=Conv2D(x, self.conv_W, self.conv_b)
        
        # ReLU activation 
        a=ReLU(z)
        
        # maxpooling layer
        m=maxpooling(a)
        
        # flatten 
        m=m.view(m.shape[0],-1)
        
        # fully connected layer
        z = th.mm(m, self.W) + self.b
        #########################################
        return z

    # ----------------------------------------------
    def train(self, loader, n_steps=10,alpha=0.01):
        """train the model 
              Input:
                loader: dataset loader, which loads one batch of dataset at a time.
                n_steps: the number of batches of data to train, an integer scalar
                alpha: the learning rate for SGD(stochastic gradient descent), a float scalar
        """
        # create a SGD optimizer
        optimizer = SGD([self.conv_W,self.conv_b,self.W,self.b], lr=alpha)
        count = 0
        while True:
            # use loader to load one batch of training data
            for x,y in loader:
                # convert data tensors into Variables
                x = Variable(x)
                y = Variable(y)
                #########################################
                ## INSERT YOUR CODE HERE

                # forward pass
                z = self.forward(x)

                # compute loss 
                L = self.compute_L(z, y)

                # backward pass: compute gradients
                self.backward(L)

                # update model parameters
                self.W.data -= alpha * self.W.grad.data
                self.b.data -= alpha * self.b.grad.data
                self.conv_W.data -= alpha * self.conv_W.grad.data
                self.conv_b.data -= alpha * self.conv_b.grad.data

                # reset the gradients 
                self.W.grad.data.zero_()
                self.b.grad.data.zero_()
                self.conv_W.grad.data.zero_()
                self.conv_b.grad.data.zero_()
                #########################################
                count+=1
                if count >=n_steps:
                    return 

