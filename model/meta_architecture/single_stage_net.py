import chainer
from model.meta_architecture.utils import concat_features


class SingleStageNet(chainer.Chain):
    def __init__(self, extractor, estimator):
        super(SingleStageNet, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.estimator = estimator

    def forward(self, x):
        h = self.extractor(x, self.extractor.extract_layers)
        h = concat_features(h, self.extractor.backbone)
        h = self.estimator(h)
        return h

    def to_gpu(self, device=None):
        super(SingleStageNet, self).to_gpu(device)
        self.extractor.to_gpu(device)
        self.estimator.to_gpu(device)

    def to(self, device):
        if device >= 0:
            self.to_gpu(device)
