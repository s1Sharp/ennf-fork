import numpy as np

def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.

    Arguments:
    x -- Array of shape (f, f)

    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    mask = np.zeros_like(x)
    ### START CODE HERE ### (≈1 line)
    mask.reshape(-1)[np.argmax(x.reshape(-1))] = 1.
    ### END CODE HERE ###

    return mask