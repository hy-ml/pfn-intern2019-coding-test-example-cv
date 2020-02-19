import chainer
import chainer.functions as F
from chainer import backend
import cupy


def build_loss(cfg, device=0):
    """ Build loss.
    Args:
        cfg (dict): Configure of key-points detection.
        device (int): Device ID.

    Returns:
        object: Loss class.

    """
    if cfg['loss'] == 'WeightedMAE':
        loss = WeightedMAE(w0=cfg['loss_w0'], w1=cfg['loss_w1'], device=device)
    elif cfg['loss'] == 'GaussianWeightedMAE':
        loss = GaussianWeightedMAE(w0=cfg['loss_w0'], w1=cfg['loss_w1'],
                                   device=device)
    elif cfg['loss'] == 'GaussianWeightedMarginMAE':
        loss = GaussianWeightedMarginMAE(w0=cfg['loss_w0'], w1=cfg['loss_w1'],
                                         device=device)
    else:
        msg = ('Not support loss: `{}`.'.format(cfg['loss']))
        raise NotImplementedError(msg)

    return loss


class Loss(object):
    """ Base class of all Loss classes.
    """
    def __init__(self, device):
        if device is None:
            device = 0
        self.device = device

    def to_gpu(self, device):
        self.device = device


class WeightedMAE(Loss):
    def __init__(self, w0=0.001, w1=1, device=None):
        super(WeightedMAE, self).__init__(device)
        self.w0 = w0
        self.w1 = w1

    def __call__(self, lst_x, y):
        if not isinstance(lst_x, list):
            lst_x = [lst_x]

        xp = backend.get_array_module(lst_x[0])

        if self.device >= 0:
            with cupy.cuda.Device(self.device):
                w_map = xp.where(y, self.w0, self.w1).astype(xp.float32)
        else:
            w_map = xp.where(y, self.w0, self.w1).astype(xp.float32)
        w_map = chainer.Variable(w_map)

        loss = 0
        for x in lst_x:
            x = x * w_map
            y = y * w_map
            loss += F.mean_absolute_error(x, y)
        return loss


class GaussianWeightedMAE(Loss):
    def __init__(self, w0=0.001, w1=1, device=None):
        super(GaussianWeightedMAE, self).__init__(device)
        self.w0 = w0
        self.w1 = w1

    def __call__(self, lst_x, y):
        if not isinstance(lst_x, list):
            lst_x = [lst_x]

        xp = backend.get_array_module(lst_x[0])

        if self.device >= 0:
            with cupy.cuda.Device(self.device):
                w_map = xp.where(self.w1 * y < self.w0, self.w0, self.w1 * y)
        else:
            w_map = xp.where(self.w1 * y < self.w0, self.w0, self.w1 * y)
        w_map = chainer.Variable(w_map)

        loss = 0
        for x in lst_x:
            x = x * w_map
            y = y * w_map
            loss += F.mean_absolute_error(x, y)

        return loss


class GaussianWeightedMarginMAE(Loss):
    def __init__(self, w0=0.001, w1=1, device=None):
        super(GaussianWeightedMarginMAE, self).__init__(device)
        self.w0 = w0
        self.w1 = w1

    def __call__(self, lst_x, y):
        if not isinstance(lst_x, list):
            lst_x = [lst_x]

        xp = backend.get_array_module(lst_x[0])

        if self.device >= 0:
            with cupy.cuda.Device(self.device):
                w_map = xp.where(self.w1 * y < self.w0, self.w0, self.w1 * y)
        else:
            w_map = xp.where(self.w1 * y < self.w0, self.w0, self.w1 * y)

        w_map = chainer.Variable(w_map)

        loss = 0
        for x in lst_x:
            x_margin = xp.where(w_map == self.w0, cupy.min(0, x), x)
            x = x * w_map
            y_margin = y_margin * w_map
            loss += F.mean_absolute_error(x, y_margin)

        return loss
