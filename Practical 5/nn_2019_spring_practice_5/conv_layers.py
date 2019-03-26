from builtins import range
import numpy as np


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # Extract the dimensions of the input data
    N,C,H,W=x.shape
    # Extract the dimensions of the Filter weights (Channels can be ignored)
    F,_,HH,WW=w.shape
    # Retrieve the stride and pad parameters from the dictionary
    stride=conv_param['stride']
    padding=conv_param['pad']
    # Compute the output volume size of the hidden layer and the weights
    out_H=1+((H+2*padding-HH)/stride)
    out_W=1+((H+2*padding-WW)/stride)
    # Initialize the output container
    out = np.zeros((N,F,out_H,out_W)) 
    
    # Pad the input data starting at position 1,1 and using constant values of 0
    input_pad=np.zeros((N,C,H+2*padding,W+2*padding))
    for i in range(N):
        for j in range(C):
            input_pad[i,j]=np.pad(x[i,j],(1,1),'constant',constant_values=(0,0))
    
    # Loop through the layer and apply the filter
    for i in range(N):
        for j in range(out_H):
            for k in range(out_W):
                for l in range(F):
                    # Extract the Input values to be convoluted
                    x_now=input_pad[i,:,j*stride:j*stride+HH,k*stride:k*stride+WW]
                    # Retrieve the indexed filter
                    kernel=w[l]
                    # Convolve the input data using the indexed filter
                    out[i,l,j,k]=np.sum(x_now*kernel)
                # Once convolution is done, add the bias    
                out[i,:,j,k]=out[i,:,j,k]+b 
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x,w,b,conv_param=cache
    # Retrieve the stride and pad parameters from the dictionary
    stride=conv_param['stride']
    padding=conv_param['pad']
    # Extract the dimensions of the input data
    N,C,H,W=x.shape
    # Extract the dimensions of the Filter weights (Channels can be ignored)
    F,_,HH,WW=w.shape
    _,_,out_H,out_W=dout.shape
    
    # Pad the input data starting at position 1,1 and using constant values of 0
    input_pad=np.zeros((N,C,H+2*padding,W+2*padding))
    for i in range(N):
        for j in range(C):
            input_pad[i,j]=np.pad(x[i,j],(1,1),'constant',constant_values=(0,0))
    
    # Store the derivatives of the biases using the right indices      
    db=np.zeros((F))
    for i in range(N):
        for j in range(out_H):
            for k in range(out_W):
                db=db+dout[i,:,j,k]
    
    # Store the derivatives of the weights and input data using the right 
    dw=np.zeros(w.shape)
    dx_pad=np.zeros(input_pad.shape)
    for i in range(N):
        for j in range(F):
            for k in range(out_H):
                for l in range(out_W):
                    # Extract the Input values that were convoluted
                    x_now=input_pad[i,:,k*stride:k*stride+HH,l*stride:l*stride+WW]
                    # Update the derivatives of the weights
                    dw[j]=dw[j]+dout[i,j,k,l]*x_now
                    # Update the derivates of the padded input
                    dx_pad[i,:,k*stride:k*stride+HH,l*stride:l*stride+WW]+=w[j]*dout[i,j,k,l]
    # Update dx  
    dx=dx_pad[:,:,1:H+1,1:W+1]                
                    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    # Extract the dimensions of the input data
    N,C,H,W=x.shape
    # Obtain pool height, width & stride from dictionary
    height_pool=pool_param['pool_height']   
    width_pool=pool_param['pool_width']
    stride=pool_param['stride']
    # Calculate the height & width of the output from Maxpooling & initialize container for the output
    out_H=1+(H-height_pool)/stride
    out_W=1+(W-width_pool)/stride
    out=np.zeros((N,C,out_H,out_W))
    
    # Perform Maxpooling by looping through the input and finding the max values within the maxpool kernel
    for i in range(N):
        for j in range(C):
            for k in range(out_H):
                for l in range(out_W):
                    out[i,j,k,l]=np.max(x[i,j,k*stride:k*stride+height_pool,l*stride:l*stride+width_pool])
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    # Store input and pool parameters from cache
    x,pool_param=cache
    # Obtain pool height, width & stride from dictionary
    height_pool=pool_param['pool_height']
    width_pool=pool_param['pool_width']  
    stride=pool_param['stride']
    # Set the dimensions based out the dout shape
    N,C,out_H,out_W=dout.shape
    
    # Initialize the derivative of x container with the shape of the input
    dx=np.zeros(x.shape)
    
    # Loop through the indices and obtain the max dx values
    for i in range(N):
        for j in range(C):
            for k in range(out_H):
                for l in range(out_W):
                    x_now=x[i,j,k*stride:k*stride+height_pool, l*stride:l*stride+width_pool]
                    x_now_max=np.max(x_now)
                    
                    # Hacked way to find the max values and update dx. Not very clean but made sense to me :)
                    for (m,n) in [(m,n) for m in range(height_pool) for n in range(width_pool)]:
                        if x_now[m,n]==x_now_max:
                            dx[i,j,k*stride+m,l*stride+n]+=dout[i,j,k,l]
                    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

