from typing import Union, Tuple
import numpy as np

from nn_lib import Tensor
from nn_lib.math_fns import Log, Exp, MatMul, SumReduce, Max, Min


def maximum(x: Tensor, y: Tensor) -> Tensor:
    result = Tensor.apply_fn(Max, x, y)
    return result


def minimum(x: Tensor, y: Tensor) -> Tensor:
    result = Tensor.apply_fn(Min, x, y)
    return result


def log(x: Tensor) -> Tensor:
    result = Tensor.apply_fn(Log, x)
    return result


def exp(x: Tensor) -> Tensor:
    result = Tensor.apply_fn(Exp, x)
    return result


def mat_mul(a: Tensor, b: Tensor) -> Tensor:
    result = Tensor.apply_fn(MatMul, a, b)
    return result


def sigmoid(x: Tensor) -> Tensor:
    result = Tensor(1) / (Tensor(1) + exp(-x))
    return result


def relu(x: Tensor) -> Tensor:
    result = maximum(x, Tensor(0))
    return result

def clip(x: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    clip_from_below = maximum(lower, x)
    clip_from_above = minimum(clip_from_below, upper)
    return clip_from_above


def reduce(x: Tensor, axis: Union[int, Tuple[int, ...], None] = None, reduction: str = 'mean', keepdims: bool = False) -> Tensor:
    """
    Apply reduction to a Tensor
    :param x: Tensor to be reduced
    :param axis: one or multiple axes to be reduced; if None reduces across all axes
    :param reduction: either 'sum' or 'mean'
    :return: reduced Tensor
    """
    assert reduction in ('mean', 'sum')

    shape = x.data.shape
    if axis is None:
        axis = tuple(range(len(shape)))
    if isinstance(axis, int):
        axis = (axis,)

    # first reduce by summation
    result = Tensor.apply_fn(SumReduce, x, axis=axis, keepdims=keepdims)
    if reduction == 'mean':
        # if reduction is 'mean' divide result by the total size of reduced axes
        denominator = np.prod(tuple(map(lambda i: shape[i], axis)))
        result = result / Tensor(denominator)
    return result



def softmax(x: Tensor) -> Tensor:
    return exp(x) / reduce(exp(x), axis=0, reduction='sum')