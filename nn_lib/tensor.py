from __future__ import annotations
from nn_lib import SetGrad
from typing import Union, Type, Iterable, Tuple
import numpy as np

from nn_lib.math_fns import Function, Add, Mul, Neg, Inv, Slice



class Tensor:
    """
    Tensor is the core class of nn_lib describing a data array that intrinsically tracks a computation tree allowing
    for computation of gradients
    """
    def __init__(self, data: Union[np.ndarray, Iterable, float, int], requires_grad: bool = False):
        """
        Create a Tensor
        :param data: data to initialize the Tensor; can be int, float, numpy array or other iterable
        :param requires_grad: whether to compute gradients for this Tensor
        """
        self.data = np.array(data, np.float32)
        self.requires_grad = requires_grad
        # an accumulated gradient value
        self.grad = None    # type: Union[None, Tensor]
        # grad_fn is a reference to the Function object that this Tensor was the result of, used for computing gradient
        # Note: Tensor can be treated as a node in a computation tree and grad_fn Function as its edges
        # Note: this is similar to how gradient tracking is done in PyTorch but still differs quite a lot
        self.grad_fn = None    # type: Union[None, Function]

    def backward(self, gradient: Union[None, Tensor] = None) -> None:
        """
        A method that computes and accumulates the gradient for the Tensor it is called for & all Tensor in its subtree

        When training a neural network the loss Tensor is a root of the computation tree since it is the last
        Tensor we computed. In order to update the parameters of the neural network we need to compute gradients of the
        Loss over each Tensor that participated in the computation, i.e. dL/dT. For this we call .backward() for the
        loss Tensor.
        :param gradient: a gradient of the tree root over the current Tensor; if self is the tree root, then None can
        be provided as gradient and gradient will be treated as equal to 1
        :return: None
        """
        # a Tensor does not require a gradient --> nothing to do
        if not self.requires_grad:
            return
        # gradient equal to None is a shortcut for gradient full of ones
        if gradient is None:
            gradient = Tensor(np.ones_like(self.data))
        # if this is the first gradient we computed for this Tensor, we need to initialize it as zeroes
        if self.grad is None:
            self.grad = Tensor(np.zeros_like(gradient.data))
        # accumulate gradient for the Tensor; accumulation allows to perform a single optimizer step for
        # multiple backward passes
        self.grad += gradient

        # if the tensor was computed as a result of some operation, compute grads for the arguments of this operation
        if self.grad_fn is not None:
            # compute the gradient over arguments of the Function that the current Tensor was result of (i.e. for the
            # children of the current Tensor)
            arg_grads = self.grad_fn.backward(gradient.data)
            # call .backward() for the child Tensors
            for arg, arg_grad in zip(self.grad_fn.args, arg_grads):
                arg.backward(Tensor(arg_grad))

    def zero_grad(self) -> None:
        """
        Reset internal gradient to zero
        :return: None
        """
        if self.grad is not None:
            self.grad = Tensor(np.zeros_like(self.grad.data))

    @staticmethod
    def apply_fn(fn_type: Type[Function], *args: Tensor, **kwargs):
        """
        Applies any Function to one or more Tensors arguments and some additional keyword arguments
        A method that is called whenever some Tensor is a result of some Function
        :param fn_type: type of function to apply
        :param args: Tensor arguments of a function, e.g. for addition c=a+b, args is (a,b)
        :param kwargs: additional non-Tensor arguments must be provided as keyword arguments, e.g. a reduction axis
        :return: a Tensor representing the result of the function
        """
        # create a Function instance
        fn = fn_type(*args, **kwargs)
        # compute the raw data result of the Function
        result_data = fn.forward()  # type: np.ndarray
        # the resulting Tensor requires gradient computation if any of the arguments that participated in its
        # computation require gradient, otherwise it would not be possible for gradient to flow to these arguments
        result_requires_grad = False
        if SetGrad.is_grad_enabled():
            result_requires_grad = any(map(lambda arg: arg.requires_grad, args))

        result = Tensor(result_data, requires_grad=result_requires_grad)
        if result_requires_grad:
            result.grad_fn = fn
        return result

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    def __str__(self) -> str:
        return f'Tensor: {np.array_str(self.data, precision=7)}'

    # Below are methods for overloading some Tensor operations like +, -, *, etc.

    def __add__(self, other: Tensor) -> Tensor:
        result = Tensor.apply_fn(Add, self, other)
        return result

    def __mul__(self, other: Tensor) -> Tensor:
        result = Tensor.apply_fn(Mul, self, other)
        return result

    def __neg__(self) -> Tensor:
        result = Tensor.apply_fn(Neg, self)
        return result

    def __sub__(self, other) -> Tensor:
        result = Tensor.apply_fn(Add, self, Tensor.apply_fn(Neg, other))
        return result

    def __truediv__(self, other: Tensor) -> Tensor:
        result = Tensor.apply_fn(Mul, self, Tensor.apply_fn(Inv, other))
        return result

    def __getitem__(self, item) -> Tensor:
        result = Tensor.apply_fn(Slice, self, slice_obj=item)
        return result

