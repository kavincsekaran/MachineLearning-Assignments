#-------------------------------------------------------------------------
# Note: please don't use any additional package except the following packages
import numpy as np
from scipy.special import psi
#-------------------------------------------------------------------------
'''
    Problem 3: LDA (Latent Dirichlet Allocation) using Variational EM method
    In this problem, you will implement the Latent Dirichlet Allocation to model text data.
    You could test the correctness of your code by typing `nosetests -v test3.py` in the terminal.

'''


#--------------------------
def variational_inference(w,beta, alpha=1., n_iter=100):
    '''
        Given a document, find the optimal values of the variational parameters: gamma and phi, using mean-field variational inference.
        Input:
            w:  the vector word ids of a document, a float numpy vector of length n. 
                n: the number of words in the document, an integer scalar.
            beta: the current estimation of parameters for word distribution on k topics, a numpy float matrix of shape k by p. 
                p: the number of all possible words (the size of the vocabulary), an integer scalar.
                k: the number of topics, an integer scalar
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
            alpha: the parameter of the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
                    Here we are assuming the parameters in all dimensions to have the same value.
            n_iter: the number of iterations for iteratively updating gamma and phi. 
        Output:
            gamma:  the optimal value for gamma, a numpy float vector of length k. 
            phi:  the optimal values for phi, a numpy float matrix of shape n by k.
                Here k is the number of topics.
        Hint: you could use the psi() in scipy package to compute digamma function
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    K=beta.shape[0]
    N=len(w)

    phi=(1/float(K))*np.ones((N, K))
    gamma=(np.ones(K)*alpha)+(N/float(K))
    for _ in range(n_iter):
        for n in range(N):
            for i in range(K):
                phi[n, i]=beta[i, w[n]]*np.exp(psi(gamma[i]))
            phi[n,:]=phi[n,:]/np.sum(phi[n,:])
            
        gamma=alpha+np.sum(phi, axis=0)

    #########################################
    return gamma, phi

#--------------------------
def E_step(W,beta, alpha=1., n_iter=100):
    '''
        Infer the optimal values for variational parameters on all documents: phi and gamma.
        Input:
            W:  the document matrix, a float numpy matrix of shape m by n. 
                Here m is the number of text documents, an integer scalar.
                n: the number of words in each document, an integer scalar.
            beta: the current estimation of parameters for word distribution on k topics, a numpy float matrix of shape k by p. 
                p: the number of all possible words (the size of the vocabulary), an integer scalar.
                k: the number of topics, an integer scalar
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
            alpha: the parameter of the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
                    Here we are assuming the parameters in all dimensions to have the same value.
            n_iter: the number of iterations for iteratively updating gamma and phi. 
        Output:
            gamma:  the optimal gamma values for all documents, a numpy float matrix of shape m by k. 
            phi:  the optimal phi values for all documents, a numpy float tensor of shape m by n by k. 
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    gamma=[]
    phi=[]
    for w in W:
        g, p = variational_inference(w, beta, alpha, n_iter)
        gamma.append(g)
        phi.append(p)

    gamma=np.vstack(gamma)
    phi=np.rollaxis(np.dstack(phi),-1)
    #########################################
    return gamma, phi 



#--------------------------
def update_beta(W, phi,p):
    '''
        update beta based upon the new values of the variational parameters. 
        Input:
            W:  the document matrix, a float numpy matrix of shape m by n. 
                Here m is the number of text documents, an integer scalar.
            phi:  the optimal phi values for all documents, a numpy float tensor of shape m by n by k. 
            p: the number of all possible words in the vocabulary.
        Output:
            beta: the updated estimation of parameters for word distribution on k topics, a numpy float matrix of shape k by p. 
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    M=phi.shape[0]
    N=phi.shape[1]
    K=phi.shape[2]
    beta=np.zeros((K, p))
    
    
    for i in range(K):
        for j in range(p):
            topic_sum=0.
            for m in range(M):
                for n in range(N):
                    w_j_dn=  1. if W[m, n]==j else 0.
                    topic_sum+=phi[m, n, i]*w_j_dn
            beta[i, j]=topic_sum
        beta[i,:] = beta[i,:] / np.sum(beta[i,:])
    
    #########################################
    return beta 

#--------------------------
def M_step(W,phi,p):
    '''
        M step of the EM algorithm. For simplicity, we ignore the updating process of alpha.
        Input:
            W:  the document matrix, a float numpy matrix of shape m by n. 
                Here m is the number of text documents, an integer scalar.
            phi:  the optimal phi values for all documents, a numpy float tensor of shape m by n by k. 
            p: the number of all possible words in the vocabulary.
        Output:
            beta: the updated estimation of parameters for word distribution on k topics, a numpy float matrix of shape k by p. 
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
    '''
    beta = update_beta(W,phi,p)
    return beta 



#--------------------------
def EM(W,k=3,p=100,alpha=1.,n_iter_var=100,n_iter_em=10):
    '''
        Variational EM algorithm for LDA. For simplicity, we ignore the updating process of alpha.
        Input:
            W:  the document matrix, a float numpy matrix of shape m by n. 
                Here m is the number of text documents, an integer scalar.
            p: the number of all possible words (the size of the vocabulary), an integer scalar.
            k: the number of topics, an integer scalar
            alpha: the parameter of the Dirichlet distribution for generating topic-mixture for each document, a float scalar (alpha>0).
            n_iter_var: the number of iterations in variational inference (E-step). 
            n_iter_em: the number of iterations in EM
        Output:
            beta: the updated estimation of parameters for word distribution on k topics, a numpy float matrix of shape k by p. 
                Each element beta[i] represents the vector of word probabilitis in the i-th topic. 
            gamma:  the optimal value for gamma, a numpy float vector of length k. 
            phi:  the optimal values for phi, a numpy float matrix of shape n by k.
    '''
    # initialize beta (for testing purpose)
    
    beta = np.arange(float(k*p)).reshape((k,p))+1.
    for i in xrange(k):
        beta[i] = beta[i] /sum(beta[i])

    for _ in xrange(n_iter_em):
        #########################################
        ## INSERT YOUR CODE HERE
        gamma, phi = E_step(W,beta, alpha=1., n_iter=100)
        beta= M_step(W,phi,p)

        #########################################
    return beta,gamma, phi 




