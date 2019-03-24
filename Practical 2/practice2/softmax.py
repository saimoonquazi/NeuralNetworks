import numpy as np
from random import shuffle
from past.builtins import xrange


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

    for i, x in enumerate(X):
        #############################################################################
        # TODO: Compute the softmax loss using explicit loops and store the result  #
        # in loss. If you are not careful here, it is easy to run into numeric      #
        # instability.                                                              #
        #############################################################################
        # Calculate the scores from the X data and the weights using i indices from loop
        z = X[i].dot(W)
        # Subtract the score from a constant value to ensure that the exponential cal-
        # culations don't run into trouble
        z -= np.max(z)
        # Calculate the softmax probablilties         
        p = np.exp(z)/(np.sum(np.exp(z)))
        
        # Calculate the loss using the sum of the log of the probabilities
        loss += -np.log(p[y[i]])
        
        #############################################################################
        # TODO: Compute the gradient using explicit loops and store it in dW.       #
        #############################################################################
        # For loop to loop through all the classes
        for k in range(W.shape[1]):
        # Computing gradient by implementing ((p-y)*x)    
            dW[:,k] += (p[k] - (k==y[i]))*X[i]
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

    loss /= X.shape[0]
    dW /= X.shape[0]

    # Add regularization to the loss and gradients.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

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
    # here, it is easy to run into numeric instability.                         #
    #############################################################################
    # Calculate the scores from the X data and the weights (vectorized)
    z=X.dot(W)
    
    # Subtract the score from a constant value to ensure that the exponential cal-
    # culations don't run into trouble. The maximum is selected over multiple axes
    # and the keepdims parameter is set to True to ensure that the result is broad-
    # casted correctly against the input.
    z -= np.max(z, axis=1, keepdims=True)
    
    # Calculate the softmax probablilties. the sum is calculated over axis 1 and 
    # the keepdims parameter is set to True to ensure that the result is broadcasted
    # correctly against the input.
    p = np.exp(z)/np.sum(np.exp(z), axis=1, keepdims=True)

    # Calculate the loss using the sum of the log of the probabilities, considering
    # the vector of stacked correct probabilites. 
    loss= np.mean(-np.log(p[range(X.shape[0]),y]))
    
    #Perform p-y in a vectorized form
    p[range(X.shape[0]),y]-=1
    # Calculate the gradient using the 
    dW=X.T.dot(p)
    # Normalize the gradient
    dW /= X.shape[0]
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    # Add regularization to the loss and gradients.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    return loss, dW
