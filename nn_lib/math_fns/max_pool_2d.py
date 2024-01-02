from typing import Tuple
import numpy as np

from nn_lib.math_fns.function import Function
from nn_lib.support_func.mask import create_mask_from_window

#https://leimao.github.io/blog/Max-Pooling-Backpropagation/

class MaxPool2d(Function):
    mode = 'max'
    """
    max pooling of activation map
    """

#    @numba.jit
    def forward(self) -> np.ndarray:
        """
        Add two arguments and return their sum

        Note: the result can have different shape because of numpy broadcasting
        https://numpy.org/doc/stable/user/basics.broadcasting.html
        :return: sum of the two arguments
        """

        #(A_prev, hparameters, mode = "max"):
        """
        Implements the forward pass of the pooling layer

        Arguments:
        A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        hparameters -- python dictionary containing "f" and "stride"
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

        Returns:
        A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
        """
        # Retrieve dimensions from the input shape
        (m, n_H_prev, n_W_prev, n_C_prev) = self.args[0].data.shape

        # Retrieve hyperparameters from "hparameters"
        stride = self.args[1].data.item()
        sliding_window_size = self.args[2].data.item()


        # Define the dimensions of the output
        n_H = int(1 + (n_H_prev - sliding_window_size) / stride)
        n_W = int(1 + (n_W_prev - sliding_window_size) / stride)
        n_C = n_C_prev

        # Initialize output matrix A
        A = np.zeros((m, n_H, n_W, n_C))

        ### START CODE HERE ###
        for i in range(m):  # loop over the training examples
            for h in range(n_H):  # loop on the vertical axis of the output volume
                for w in range(n_W):  # loop on the horizontal axis of the output volume
                    for c in range(n_C):  # loop over the channels of the output volume

                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * stride
                        vert_end = vert_start + sliding_window_size
                        horiz_start = w * stride
                        horiz_end = horiz_start + sliding_window_size

                        # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                        a_prev_slice = self.args[0].data[i, vert_start:vert_end, horiz_start:horiz_end, c]

                        # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                        #if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                        '''
                        elif mode == "average":
                            A[i, h, w, c] = np.mean(a_prev_slice)
                        '''
        ### END CODE HERE ###

        # Making sure your output shape is correct
        assert (A.shape == (m, n_H, n_W, n_C))

        return A
#   @numba.jit
    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Note: because of the broadcasting, arguments and grad_output can have different shapes, we reduce the
        gradients to their original shape inside reduce_gradient() parent method, hence it is ok here for the
        resulting gradients to have shapes different from the original arguments
        :param grad_output: gradient over the result of the addition operation
        :return: a tuple of gradients over two addition arguments
        """

        """
    
        Implements the backward pass of the pooling layer

        Arguments:
        dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
        cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters
        mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

        Returns:
        dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
        """

        ### START CODE HERE ###

        stride = self.args[1].data.item()
        sliding_window_size = self.args[2].data.item()
        # Retrieve dimensions from A_prev's shape and dA's shape (≈2 lines)
        m, n_H_prev, n_W_prev, n_C_prev = self.args[0].data.shape
        m, n_H, n_W, n_C = grad_output.shape

        # Initialize dA_prev with zeros (≈1 line)
        dA_prev = np.zeros(self.args[0].data.shape)

        for i in range(m):  # loop over the training examples
            # select training example from A_prev (≈1 line)
            a_prev = self.args[0].data[i]
            for h in range(n_H):  # loop on the vertical axis
                for w in range(n_W):  # loop on the horizontal axis
                    for c in range(n_C):  # loop over the channels (depth)
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * stride
                        vert_end = vert_start + sliding_window_size
                        horiz_start = w * stride
                        horiz_end = horiz_start + sliding_window_size

                        # Compute the backward propagation in both modes.
                        #if mode == "max":
                        # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice (≈1 line)
                        mask = create_mask_from_window(a_prev_slice)
                        # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += np.multiply(mask, grad_output[i, h, w, c])
                        '''
                        elif mode == "average":
                            # Get the value a from dA (≈1 line)
                            da = dA[i, h, w, c]
                            # Define the shape of the filter as fxf (≈1 line)
                            shape = (f, f)
                            # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
                            dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(da, shape)
                        '''
        ### END CODE ###

        # Making sure your output shape is correct
        assert (dA_prev.shape == self.args[0].data.shape)

        return tuple([dA_prev])
