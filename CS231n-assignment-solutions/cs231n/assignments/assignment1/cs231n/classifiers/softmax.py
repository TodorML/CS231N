from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
import math

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(N):
      score = X[i].dot(W)
      correct_label = y[i]
      exp_score = np.exp(score)
      loss -= np.log(exp_score[correct_label]/np.sum(exp_score))  # np.log is natural log
      softmax_prob = exp_score/np.sum(exp_score) # p
      Y_true_class = np.zeros(W.shape[1])
      Y_true_class[y[i]]=1
      dW += np.outer(X[i].T,softmax_prob-Y_true_class)


    loss = loss/N
    loss = loss + reg*np.sum(np.square(W))
    dW = dW/N
    dW = dW + reg*W*2
    

    

      


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    score_matrix = X @ W
    exp_score_matrix = np.exp(score_matrix)
    sum_exp_matrix = np.sum(exp_score_matrix, axis=1)
    select_true_labels = exp_score_matrix[np.arange(len(y)),y]
    loss=np.sum(-np.log(select_true_labels/sum_exp_matrix ))
    loss = loss/len(y)
    loss += reg*np.sum(W**2)
    softmax_prob_matrix = exp_score_matrix/sum_exp_matrix[:,np.newaxis]  # P
    Y_true_classes = np.zeros(softmax_prob_matrix.shape)
    Y_true_classes[np.arange(len(y)),y]=1
    dW =  X.T @ (softmax_prob_matrix-Y_true_classes)
    dW = dW/len(y)
    dW += reg*W*2
    








    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
