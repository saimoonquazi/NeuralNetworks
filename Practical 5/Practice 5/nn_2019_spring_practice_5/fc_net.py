from builtins import range
from builtins import object
import numpy as np

from layers import *
from layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        # See also: http://cs231n.github.io/neural-networks-2/#init                #
        ############################################################################
        # Initialize the weights in the self.params dictionary, with random numbers 
        # multiplied by the weight scales. Dimensions should follow network layer 
        # dimensions, i.e. first layer dim [input x hidden], second layer [hiddenxoutput]
        self.params["W1"]= weight_scale*np.random.randn(input_dim,hidden_dim)
        self.params["W2"]=weight_scale*np.random.randn(hidden_dim,num_classes)
        #Initialize the biases with 0s corresponding to the right vector dimensions
        self.params["b1"]=np.zeros(hidden_dim)
        self.params["b2"]=np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        #Initialize Weights and Biases for use in the functions below 
        W1 = self.params["W1"]
        W2 = self.params["W2"]
        b1 = self.params["b1"]
        b2 = self.params["b2"]
        
        #Perform forward pass with Relu Activation for the hidden layer
        h, cachel1 =affine_relu_forward(X,W1,b1)
        #Use output of hidden layer to complete the forward pass
        scores, cachel2=affine_forward(h,W2,b2)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # Evaluate the Softmax loss and derivative of scores
        loss,dscores =softmax_loss(scores,y)
        # Apply Regularization to the loss 
        loss += 0.5*self.reg*(np.sum(W1**2)+np.sum(W2**2))
        # Use the Affine_backward module to compute backward pass of affine layer
        dh,grads["W2"],grads["b2"]=affine_backward(dscores,cachel2)
        # Use the Affine_relu_backward to compute the backward pass for a layer of 
        # rectified linear units (ReLUs).
        dx,grads["W1"],grads["b1"]=affine_relu_backward(dh,cachel1)
        # Apply Regularization to the Weight gradients
        grads["W2"]+=W2*self.reg
        grads["W1"]+=W1*self.reg
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        # Find random weights (with mean 0) for the first layer based on the scale 
        # defined while maintianing the right dimensions
        W1_layer1= np.random.normal(0,weight_scale,input_dim*hidden_dims[0])
        # Store first layer weights in teh params dictionary, reshaping just to be sure
        self.params["W1"]= W1_layer1.reshape((input_dim,hidden_dims[0]))
        # Initialize first layer biases with zeros 
        self.params['b1'] = np.zeros((hidden_dims[0]))
        
        # Loop through initialization for intermediary layers
        for i in xrange(1,self.num_layers-1):
            # Initialize the indexed weights in the dictionary with random numbers with mean 0 and defined weight scale, 
            #trying to maintain dimensionality
            self.params['W'+str(i+1)]=np.random.normal(0,weight_scale,hidden_dims[i-1]*hidden_dims[i]).reshape(              (hidden_dims[i-1],hidden_dims[i]))
            # Initialize the indexed biases in the dictionary with zeros
            self.params['b' + str(i + 1)] = np.zeros((hidden_dims[i]))     
        
        # Initialize the final layer weights with random numbers with mean 0 and defined weight scale.
        self.params['W'+str(self.num_layers)]=np.random.normal(0,weight_scale,hidden_dims[-1]*num_classes).reshape(              (hidden_dims[-1],num_classes))
        # Initialize the final layer biases in the dictionary with zeros
        self.params['b' + str(self.num_layers)] = np.zeros((num_classes))
        
        #If batchnorm is being used, initialize the relevant parameters (gamma and beta).
        # Maintain the right dimensionality
        if self.use_batchnorm:
            for i in xrange(1,self.num_layers):
                self.params['beta'+str(i)]=np.zeros(hidden_dims[i-1])
                self.params['gamma'+str(i)]=np.ones(hidden_dims[i-1])
           
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # Container to store the cache data from each layer
        list_cache=[]
        # Make a copy of the input data to make it easier to handle during iterations
        input_data=X.copy()
        
        # If dropout is being used
        if self.use_dropout:
            # Loop through the layers 
            for i in xrange(1,self.num_layers):
                # Perfrom Affine Forward Pass
                a, fc_cache = affine_forward(input_data, self.params['W'+str(i)], self.params['b'+str(i)])
                # Perfrom Relu activation for forward pass
                r, relu_cache = relu_forward(a)
                # Use the ouptut from Relu activation to perform dropout forward pass
                dropout_output, dropout_cache = dropout_forward(r, self.dropout_param)
                # Store the cached values
                cache=(fc_cache, relu_cache, dropout_cache)
                # Store current cache to the Cache list
                list_cache.append(cache)
                # Set input data as current output for next layer
                input_data=dropout_output

            # Perfrom the affine forward pass for the last layer    
            scores,dr_cache_ln=affine_forward(input_data,self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)])
            # Store current cache to Cache list
            list_cache.append(dr_cache_ln)
        
        # If use of batchnorm is desired
        elif self.use_batchnorm:
            # Loop throught the layers
            for i in xrange(1,self.num_layers):
                # Perform Affine forward
                affine_out, fc_cache = affine_forward(input_data, self.params['W'+str(i)], self.params['b'+str(i)])
                # Perform Batchnorm first then Relu (Combo 1)
                bn_out, bn_cache = batchnorm_forward(affine_out, self.params['gamma'+str(i)], 
                                                     self.params['beta'+str(i)], self.bn_params[i-1])
                relu_out, relu_cache = relu_forward(bn_out)
                # Perform Relu activation & then Batchnorm (Combo 2) for Question 2
                #relu_out, relu_cache = relu_forward(affine_out)
                #bn_out, bn_cache = batchnorm_forward(relu_out, self.params['gamma'+str(i)], self.params['beta'+str(i)], self.bn_params[i-1])
                # Store all the cache
                batchnorm_cache = (fc_cache, bn_cache, relu_cache)
                list_cache.append(batchnorm_cache)
                input_data=relu_out
        
        
        # Otherwise run normal mode
        else:
            #Loop through the layers
            for i in xrange(1,self.num_layers):
                # Perform forward pass with Relu Activation at given indexed layer
                layer_output,layer_cache=affine_relu_forward(input_data,self.params['W'+str(i)],self.params['b'+str(i)])
                # Store current cache to the cache list
                list_cache.append(layer_cache)
                # Set input data as current output for next layer
                input_data=layer_output
            
       # Perfrom the affine forward pass for the last layer    
        scores, cache_ln=affine_forward(input_data,self.params['W'+str(self.num_layers)],self.params['b'+str(self.num_layers)])
       # Store current cache to Cache list
        list_cache.append(cache_ln)
           
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # Evaluate the Softmax loss and derivative of scores
        loss,dout=softmax_loss(scores,y)
        # Initialize containers to store Derivatives of weights and biases
        list_dw=[]
        list_db=[]
        list_dgamma=[]
        list_dbeta=[]

        # Extract and remove the last layer entry in the Cache List
        cache=list_cache.pop()
        # Perform affine backward pass on the cached data for the last layer
        dx,dw,db=affine_backward(dout,cache)
        # Store the derivative of weights and biases at index 0 of respective lists
        list_dw.insert(0,dw)
        list_db.insert(0,db)
        dout=dx        
        
        # If Dropout is being used
        if self.use_dropout:

        
            # Loop through the cached entries for all the intermediary layers
            for i in xrange(len(list_cache)):
                # Extract and remove the last entry in the cache list
                cache=list_cache.pop()
                # Extract the data from the cache file
                fc_cache, relu_cache, dropout_cache = cache
                # Perform dropout backward pass
                dd = dropout_backward(dout, dropout_cache)
                # Perfrom Relu activation for backward pass  
                dr = relu_backward(dd, relu_cache)
                # Perfrom Affine backward Pass
                dx, dw, db = affine_backward(dr, fc_cache)
                # Update list of derivatives of weights and biases
                list_dw.insert(0,dw)
                list_db.insert(0,db)
                # Set derivative of output as derivative of x
                dout=dx
                
            para_loss=0
            
            # Loop through the values in list of derivatives of weights
            for i in xrange(len(list_dw)):
                #Apply regularization to the weights
                W=self.params['W'+str(i+1)]
                list_dw[i]+=self.reg*W
                # Use para_loss variable to store the iterative penalty terms for the regularization
                para_loss+=np.sum(W**2)
            # Regularize the loss    
            loss+=0.5*self.reg*para_loss
        
            # Loop through and update the grads dictionary entries for derivatives of weights and biases
            for i in xrange(len(list_dw)):
                grads['W'+str(i+1)]=list_dw[i]
                grads['b'+str(i+1)]=list_db[i]
        
        # If use of batchnorm is desired
        elif self.use_batchnorm:
            # Loop through the cached entries for all the intermediary layers
            for i in xrange(len(list_cache)):
                # Get the last entry from the cache list and store relevant layer caches
                cache=list_cache.pop()
                fc_cache, bn_cache, relu_cache = cache
                # Perform Relu Backward Pass then Batchnorm Backward (Combo1)
                drelu_out = relu_backward(dout, relu_cache)
                dbn_out, dgamma, dbeta = batchnorm_backward(drelu_out, bn_cache)
                dx, dw, db = affine_backward(dbn_out, fc_cache)
                #Perform Batchnorm Backward then Relu Backward Pass  (Combo2) for Question 2
                #dbn_out, dgamma, dbeta = batchnorm_backward(dout, bn_cache)
                #drelu_out = relu_backward(dbn_out, relu_cache)
                #dx, dw, db = affine_backward(drelu_out, fc_cache)
                # Store relevant gradients in the containers and update computed dx for next layer
                list_dw.insert(0,dw)
                list_db.insert(0,db)
                list_dgamma.insert(0,dgamma)
                list_dbeta.insert(0,dbeta)
                dout=dx        
            
            # Regularize the loss and compute the parametric loss for all layers
            para_loss=0
            for i in xrange(len(list_dw)):
                list_dw[i]+=self.reg*self.params['W'+str(i+1)]
                para_loss+=np.sum((self.params['W'+str(i+1)])**2)
                # Store the gradients of Weights and biases for all the layers
                grads['W'+str(i+1)]=list_dw[i]
                grads['b'+str(i+1)]=list_db[i]
                
            # Calculate the loss and store sore respective gradients of Gamma and beta    
            for i in xrange(len(list_dgamma)):
                list_dgamma[i]+=self.reg*self.params['gamma'+str(i+1)]
                list_dbeta[i]+=self.reg*self.params['beta'+str(i+1)]                    
                para_loss+=np.sum((self.params['gamma'+str(i+1)])**2)+np.sum((self.params['beta'+str(i+1)])**2)
                grads['gamma'+str(i+1)]=list_dgamma[i]
                grads['beta'+str(i+1)]=list_dbeta[i]
            
            # Put it all together to find the total loss.
            loss+=0.5 * self.reg*para_loss
        
        # If dropout is not specified, run normal mode       
        else:       
            # Loop through the cached entries for all the intermediary layers
            for i in xrange(len(list_cache)):
                # Extract and remove the last entry in the cache list                
                cache=list_cache.pop()
                # Perform Backward pass with Relu activation 
                dx,dw,db=affine_relu_backward(dout,cache)
                # Update list of derivatives of weights and biases
                list_dw.insert(0,dw)
                list_db.insert(0,db)
                # Set derivative of output as derivative of x
                dout=dx
            para_loss=0
        
            # Loop through the values in list of derivatives of weights
            for i in xrange(len(list_dw)):
                #Apply regularization to the weights                
                W=self.params['W'+str(i+1)]
                list_dw[i]+=self.reg*W
                # Use para_loss variable to store the iterative penalty terms for the regularization                
                para_loss+=np.sum(W**2)
            # Regularize the loss
            loss+=0.5*self.reg*para_loss

            # Loop through and update the grads dictionary entries for derivatives of weights and biases            
            for i in xrange(len(list_dw)):
                grads['W'+str(i+1)]=list_dw[i]
                grads['b'+str(i+1)]=list_db[i]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    affine_out, fc_cache = affine_forward(x, w, b)
    bn_out, bn_cache = batchnorm_forward(affine_out, gamma, beta, bn_param)
    relu_out, relu_cache = relu_forward(bn_out)
    cache = (fc_cache, bn_cache, relu_cache)
    return relu_out, cache


