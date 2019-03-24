from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, M)
    b1: First layer biases; has shape (M,)
    W2: Second layer weights; has shape (M, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons M in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    # Find the outputs of the first layer using the dot product of the input and 
    # the weights and adding the bias.
    A1 = X.dot(W1) + b1
    # Find the maximum value of the array holding the output from the first layer.
    # Comparing the values with 0 eliminates any values that are negative. 
    H1 = np.maximum(A1,0)
    # Compute the outputs of the second layer using the dot product of the output
    # from the first layer and adding the bias for the second layer.
    scores = H1.dot(W2)+b2       
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    # Find the exponential of the scores for the Softmax function numerator
    scores_exp = np.exp(scores)
    # Calculate the probabilites using the softmax classifier
    p=scores_exp/(np.sum(scores_exp,axis=1,keepdims=True))
    # Calculate the loss of the predictions
    loss=-np.mean(np.log(p[np.arange(N),y]))
    # Regularization the loss 
    loss+= reg*(np.sum(W1**2)+np.sum(W2**2))
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    # Calculate the derivative of the scores calulated above
    d_scores=p
    # Calculate P-Y
    d_scores[range(N),y] -= 1
    # Divide by N. Now we have (1/N*(P-Y))
    d_scores=d_scores/N
    # Dot Product the above with the H1 Transposed to complete the formula and get 
    # the gradient with respect to W2.
    grads['W2'] = H1.T.dot(d_scores)
    # Compute B2 by simply summing the d_scores calculated above. Summing performs 
    # the same operation as multiplication by the identity matrix in the formula.
    grads['b2'] = np.sum(d_scores,axis=0)
    # Compute the derivative with respect to H1
    d_H1 = d_scores.dot(W2.T)
    d_H1[A1<=0]=0
    
    # Calculate the gradients with respect to W1 using the formula X^T.d_H1
    grads['W1'] = X.T.dot(d_H1)
    # Calculate the gradients with respect to b1 
    grads['b1'] = np.sum(d_H1,axis=0)
    
    # Apply regularization ot the calculated gradients
    grads['W2'] +=reg*W2
    grads['W1'] +=reg*W1
    
    
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    
    
    for it in xrange(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      # Generate a random array for sample indices from the number of training 
      # samples
      samples = np.random.choice(num_train,batch_size)
      # Pick out the X values from those indices
      X_batch = X[samples]
      # Pick out the corresponding Y values from those indices
      y_batch = y[samples]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      #Calculate the new weights and biases based on the learning rate
      self.params['W1']+=-learning_rate*grads['W1']
      self.params['W2']+=-learning_rate*grads['W2']
      self.params['b1']+=-learning_rate*grads['b1']
      self.params['b2']+=-learning_rate*grads['b2']  
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay
    

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    # Find the outputs of the first layer using the dot product of the input and 
    # the weights and adding the bias.
    z1=X.dot(self.params['W1'])+self.params['b1']
    # Find the maximum value of the array holding the output from the first layer.
    # Comparing the values with 0 eliminates any values that are negative.
    a1=np.maximum(z1,0)
    # Compute the outputs of the second layer using the dot product of the output
    # from the first layer and adding the bias for the second layer.
    scores=a1.dot(self.params['W2'])+self.params['b2']
    # Find the index for the largest values along axis 1 in the scores matrix
    y_pred=np.argmax(scores,axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


