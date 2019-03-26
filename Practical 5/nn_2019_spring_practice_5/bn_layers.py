from builtins import range
import numpy as np

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample standard deviation
    are computed from minibatch statistics and used to normalize the incoming
    data. During training we also keep an exponentially decaying running mean of
    the mean and standard deviation of each feature, and these averages are used
    to normalize data at test-time.

    At each timestep we update the running averages for mean and standard
    deviation using an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_std = momentum * running_std + (1 - momentum) * sample_std

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and stddev for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_std Array of shape (D,) giving running stddev of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_std = bn_param.get('running_std', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and standard deviation,#
        # use these statistics to normalize the incoming data, and scale and  #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and standard          #
        # deviation together with the momentum variable to update the running #
        # mean and running standard deviation, storing your result in the     #
        # running_mean and running_std variables.                             #
        #######################################################################
        #Fint the input mean and standard deviations
        input_mean=np.mean(x,axis=0)
        input_std=np.var(x,axis=0)
        # Compute the normalized values for the input
        x_norm=(x-input_mean)/(np.sqrt(input_std+eps))
        # Scale and shift the normalized input
        print(gamma.shape)
        print(x_norm.shape)
        out=gamma*x_norm + beta
        # Compute the running mean & standard deviations using momentum
        running_mean=momentum*running_mean+(1-momentum)*input_mean
        running_std=momentum*running_std+(1-momentum)*input_std
        # Store the cache
        cache=(x,input_mean,input_std,x_norm,beta,gamma,eps)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # Use the running_mean & standard deviations computed above to normalize 
        # the input values 
        x_norm=(x-running_mean)/(np.sqrt(running_std+eps))
        # Scale and shift the normalized input
        out=gamma*x_norm +beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_std'] = running_std

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # HINT: https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html #
    ###########################################################################
    # Followed the instructions in the link from the hint. Deriving the formulas
    # backpass seemed more complex than I anticipated. 
    # Extract the variables from the cache
    (x,input_mean,input_std,x_norm,beta,gamma,eps)=cache
    # Obtain the dimensions of the input
    N,D=dout.shape
    # Calculate the derivative of beta through the summation gate
    dbeta=np.sum(dout,axis=0)
    # Calculate the derivative of gamma through the multiplication gate 
    dgamma=np.sum(dout*x_norm,axis=0)
    # Use the derivative of the normalized input & compute the derivaties of the numerator & denominator seperately
    dx_norm=gamma*dout
    d_ivar=np.sum(dx_norm*(x-input_mean),axis=0)
    dx_mu1=dx_norm*(1./np.sqrt(input_std+eps))
    dsqrtstd=-1./(np.sqrt(input_std+eps)**2)*d_ivar
    dvar=0.5*1./np.sqrt(input_std+eps)*dsqrtstd
    dsq=1./N*np.ones((N,D))*dvar
    dx_mu2=2*(x-input_mean)*dsq
    dx1=dx_mu1+dx_mu2
    # Compute the derivative of the mean
    d_mean=-1*np.sum(dx1,axis=0)
    dx2=1./N*(np.ones((N,D)))*d_mean
    # Sum up the gradients to get the final gradient
    dx=dx1+dx2
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    #                                                                         #
    # HINT: http://cthorey.github.io./backpropagation/                        #
    ###########################################################################
    # Having worked it out on paper, this was much more intuitive :D
    # Extract the variables from the cache
    (x,input_mean,input_std,x_norm,beta,gamma,eps)=cache
    # Obtain the dimensions of the input
    N=x.shape[0]
    # Compute the gradients of beta and gamma and normalzied input, same as above
    dbeta=np.sum(dout,axis=0)
    dgamma=np.sum(dout*x_norm,axis=0)
    dx_norm=gamma*dout
    # Compute the gradients of the mean and standard deviations
    dinput_std = np.sum(-1.0/2*dx_norm*x_norm/(input_std+eps), axis =0)
    dinput_mean = np.sum(-1/np.sqrt(input_std+eps)* dx_norm, axis = 0)
    #Compute the gradient with respect to the inputs
    dx = 1/np.sqrt(input_std+eps)*dx_norm + dinput_std*2.0/N*(x-input_mean) + 1.0/N*dinput_mean
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    # Get the dimesnions of dout
    N,C,W,H=x.shape
    # Perform batchnorm forward pass, transposing the input so that the right 
    # shape can then be obtained for spatial transformation
    out,cache=batchnorm_forward(x.transpose(0,3,2,1).reshape((N*H*W,C)), gamma, beta, bn_param)
    # Reshape the output back
    out=out.reshape(N,W,H,C).transpose(0,3,2,1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    # Get the dimesnions of dout
    N,C,W,H=dout.shape
    # Use the above method to compute the gradients. Key note is to transpose the 
    # input so that it can be reshaped accordingly, as above. 
    dx, dgamma, dbeta = batchnorm_backward_alt(dout.transpose(0,3,2,1).reshape((N*H*W,C)),cache)
    # Reshape the dx back accordingly.
    dx = dx.reshape(N,W,H,C).transpose(0,3,2,1)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta
