from typing import Tuple
import numpy as np

from nn_lib.math_fns.function import Function
from nn_lib.support_func.conv_step import conv_single_step
from nn_lib.support_func.padding import zero_pad

#https://tinynet.autoai.org/en/latest/induction/convolution.html

class Conv2d(Function):
    """
    Addition of two elements
    """
    def forward(self) -> np.ndarray:
        """

        Note: the result can have different shape because of numpy broadcasting
        https://numpy.org/doc/stable/user/basics.broadcasting.html
        :return: sum of the two arguments
        """
        #(A_prev, W, b, hparameters):
        """
        Implements the forward propagation for a convolution function

        Arguments:
        A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
        *deleted* b -- Biases, numpy array of shape (1, 1, 1, n_C)
        
        kwargs -- python dictionary containing "stride" and "pad"

        Returns:
        Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward() function
        """

        ### START CODE HERE ###
        # Retrieve dimensions from A_prev's shape (≈1 line)
        (m, n_H_prev, n_W_prev, n_C_prev) = self.args[0].data.shape
        W = self.args[1].data
        # Retrieve dimensions from W's shape (≈1 line)
        (f, f, n_C_prev, n_C) = W.shape

        # Retrieve information from "hparameters" (≈2 lines)
        stride = self.kwargs['stride']
        pad = self.kwargs['padding']

        # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
        n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
        n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

        # Initialize the output volume Z with zeros. (≈1 line)
        Z = np.zeros((m, n_H, n_W, n_C))

        # Create A_prev_pad by padding A_prev
        A_prev_pad = zero_pad(self.args[0].data, pad)

        for i in range(m):  # loop over the batch of training examples
            a_prev_pad = A_prev_pad[i]  # Select ith training example's padded activation
            for h in range(n_H):  # loop over vertical axis of the output volume
                for w in range(n_W):  # loop over horizontal axis of the output volume
                    for c in range(n_C):  # loop over channels (= #filters) of the output volume

                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                        Z[i, h, w, c] = conv_single_step(a_slice_prev, W[..., c])

        ### END CODE HERE ###

        # Making sure your output shape is correct
        assert (Z.shape == (m, n_H, n_W, n_C))

        # Save information in "cache" for the backprop

        return Z


    def _backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """


        Note: because of the broadcasting, arguments and grad_output can have different shapes, we reduce the
        gradients to their original shape inside reduce_gradient() parent method, hence it is ok here for the
        resulting gradients to have shapes different from the original arguments
        :param grad_output: gradient over the result of the addition operation
        :return: a tuple of gradients over two addition arguments
        """
        #(dZ, cache):
        """
        Implement the backward propagation for a convolution function

        Arguments:
        dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
        cache -- cache of values needed for the conv_backward(), output of conv_forward()

        Returns:
        dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
                   numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
        dW -- gradient of the cost with respect to the weights of the conv layer (W)
              numpy array of shape (f, f, n_C_prev, n_C)
        *deleted* db -- gradient of the cost with respect to the biases of the conv layer (b)
              numpy array of shape (1, 1, 1, n_C)
        """

        ### START CODE HERE ###
        # Retrieve information from "cache"
        A_prev = self.args[0].data
        W = self.args[1].data
        # Retrieve information from "parameters"
        stride = self.kwargs['stride']
        pad = self.kwargs['padding']

        # Retrieve dimensions from A_prev's shape
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

        # Retrieve dimensions from W's shape
        (f, f, n_C_prev, n_C) = W.shape

        # Retrieve dimensions from dZ's shape
        (m, n_H, n_W, n_C) = grad_output.shape

        # Initialize dA_prev, dW, db with the correct shapes
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
        dW = np.zeros((f, f, n_C_prev, n_C))

        # Pad A_prev and dA_prev
        A_prev_pad = zero_pad(A_prev, pad)
        dA_prev_pad = zero_pad(dA_prev, pad)

        for i in range(m):  # loop over the training examples

            # select ith training example from A_prev_pad and dA_prev_pad
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]

            for h in range(n_H):  # loop over vertical axis of the output volume
                for w in range(n_W):  # loop over horizontal axis of the output volume
                    for c in range(n_C):  # loop over the channels of the output volume

                        # Find the corners of the current "slice"
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        # Use the corners to define the slice from a_prev_pad
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * grad_output[i, h, w, c]
                        dW[:, :, :, c] += a_slice * grad_output[i, h, w, c]

            # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
            if pad > 0:
                dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
            else:
                dA_prev[i] = da_prev_pad

        ### END CODE HERE ###

        # Making sure your output shape is correct
        assert (dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))

        return tuple([dA_prev, dW])
