import sys
import numpy as np
import chainer
sys.path.append('../..')
from loss.loss import WeightedMAE


def test_weighted_mae():
    x1 = chainer.Variable(np.random.rand(2, 16, 56, 56).astype(np.float32))
    x2 = chainer.Variable(np.random.rand(2, 16, 56, 56).astype(np.float32))
    loss = WeightedMAE()
    output = loss(x1, x2)
    output.backward()
    print('Success Test: `test_weighted_mae`.')


if __name__ == '__main__':
    test_weighted_mae()

