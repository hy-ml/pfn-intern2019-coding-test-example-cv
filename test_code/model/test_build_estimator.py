import sys
import numpy as np
sys.path.append('../..')
from model.build import build_estimators


def main():
    x = np.random.rand(1, 256, 56, 56).astype(np.float32) * 100

    layers_def1 = [[
        [3, 128, 1], [3, 128, 1], [3, 128, 1], [1, 512, 0], [1, 16, 0], 's']]
    estimator = build_estimators(layers_def1)
    h = estimator[0](x)
    assert h.shape == (1, 16, 56, 56)

    layers_def2 = [[
        [3, 128, 1], [3, 128, 1], [3, 128, 1], [1, 512, 0], 'p', [1, 19, 0]]]
    estimator = build_estimators(layers_def2)
    h = estimator[0](x)
    assert h.shape == (1, 19, 28, 28)
    print('Success Test `build_estimator`.')


if __name__ == '__main__':
    main()
