from chainer import Chain
from chainer import reporter


class KeypointsDetector(Chain):
    def __init__(self, predictor, loss_fun):
        """ Train wrapper of key-points detector.

        Args:
            predictor:
            loss_fun:
        """
        super(KeypointsDetector, self).__init__()
        with self.init_scope():
            self.predictor = predictor
            self.loss_fun = loss_fun
        self.y = None
        self.loss = None

    def forward(self, x, heatmap):
        self.y = self.predictor(x)
        self.loss = self.loss_fun(self.y, heatmap)
        reporter.report({'loss': self.loss}, self)
        return self.loss

    def to_cpu(self):
        super(KeypointsDetector, self).to_cpu()

    def _to_gpu(self, device):
        self.to_gpu(device)
        self.predictor.to(device)
        self.loss_fun.to_gpu(device)

    def to(self, device=None):
        if device is None:
            device = 0

        if device >= 0:
            self._to_gpu(device)
