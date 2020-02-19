import chainer
from model.meta_architecture.utils import concat_features


class Stage3Net(chainer.Chain):
    def __init__(self, extractor, lst_estimator):
        super(Stage3Net, self).__init__()
        with self.init_scope():
            self.extractor = extractor

            self.est0 = lst_estimator[0]
            self.est1 = lst_estimator[1]
            self.est2 = lst_estimator[2]
            self.lst_estimator = [self.est0, self.est1, self.est2]

    def forward(self, x):
        # extract feature by extractor
        f_map = self.extractor(x, self.extractor.extract_layers)
        # TODO: support using feature maps from multi-layers
        # concatenate feature map when using feature maps from multi-layers
        f_map = concat_features(f_map,  self.extractor.backbone)

        lst_h = []
        for stage, estimator in enumerate(self.lst_estimator):
            # Stage 1
            if stage == 0:
                h = estimator(f_map)
                lst_h.append(h)
            # Stege 2, 3
            else:
                # concatenate feature map and heatmap of the previous stage
                f_map_with_heatmap = \
                    chainer.functions.concat((f_map, lst_h[-1]), axis=1)
                h = estimator(f_map_with_heatmap) + lst_h[-1]
                lst_h.append(h)

        return lst_h

    def to_gpu(self, device=None):
        super(Stage3Net, self).to_gpu(device)
        self.extractor.to_gpu(device)
        for estimator in self.lst_estimator:
            estimator.to_gpu(device)

    def to(self, device=None):
        if device >= 0:
            self.to_gpu(device)
