import numpy as np
import cupy as cp

import chainer
from chainer import backend
from chainer import function_node
from chainer.utils import type_check


class MarginMeanAbsoluteError(function_node.FunctionNode):

    """Mean absolute error function."""
    def __init__(self, threshold_margin):
        self.threshold_margin = threshold_margin

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x0', 'x1'))
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        diff = self.diff.ravel()
        diff_margin = np.where(x1 > self.threshold_margin, diff,
                               np.min((np.zeros_like(diff), diff)))
        return np.array(abs(diff_margin).sum() / diff_margin.size,
                        dtype=diff_margin.dtype),

    def forward_gpu(self, inputs):
        x0, x1 = inputs
        self.diff = x0 - x1
        diff = self.diff.ravel()
        diff_margin = cp.where(x1 > self.threshold_margin, diff,
                               cp.min((cp.zeros_like(diff), diff)))
        return abs(diff_margin).sum() / diff_margin.dtype.type(diff_margin.size),

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        coeff = gy * gy.data.dtype.type(1. / self.diff.size)
        coeff = chainer.functions.broadcast_to(coeff, self.diff.shape)
        gx0 = coeff * backend.get_array_module(gy.data).sign(self.diff)
        return gx0, -gx0


def mean_absolute_error(x0, x1, threshold_margin):
    """Mean absolute error function.
    This function computes mean absolute error between two variables. The mean
    is taken over the minibatch.
    Args:
        x0 (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        x1 (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
    Returns:
        ~chainer.Variable:
            A variable holding an array representing the mean absolute
            error of two inputs.
    """
    return MarginMeanAbsoluteError(threshold_margin).apply((x0, x1))[0]