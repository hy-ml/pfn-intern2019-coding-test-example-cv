import warnings
import chainer
from chainer.links import Convolution2D
from chainer.functions import max_pooling_2d, sigmoid, relu
from chainer.links.model.vision.vgg import VGG16Layers, VGG19Layers
from chainer.links.model.vision.googlenet import GoogLeNet
from chainer.links.model.vision.resnet import ResNet50Layers, ResNet101Layers,\
    ResNet152Layers
from chainercv.links import Conv2DActiv
from model.meta_architecture import SingleStageNet, Stage3Net, \
    ResidualStage3Net, SubResidualStage3Net


class MaxPool2D(object):
    def __call__(self, x):
        h = max_pooling_2d(x, ksize=2, stride=2)
        return h


def sigmoid_for_residual(x):
    return 2 * sigmoid(x) - 1


def build_extractor(cfg):
    """ Build feature extractor.

    Args:
        cfg (dict): Config of key-points detector.

    Returns:
        Chain: Feature extractor.

    """
    model = cfg['extractor']
    extract_layers = cfg['extract_layers']
    pretrained_model = 'auto'

    if model == 'vgg16':
        extractor = VGG16Layers(pretrained_model)
    elif model == 'vgg19':
        extractor = VGG19Layers(pretrained_model)
    elif model == 'resnet50':
        extractor = ResNet50Layers(pretrained_model)
    elif model == 'resnet101':
        extractor = ResNet101Layers(pretrained_model)
    elif model == 'resnet152':
        extractor = ResNet152Layers(pretrained_model)
    elif model == 'googlenet':
        extractor = GoogLeNet(pretrained_model)
    else:
        raise NotImplemented('Not support extractor: {}.'.format(model))

    extractor.backbone = model
    extractor.extract_layers = extract_layers

    return extractor


def build_estimators(cfg):
    """ Build key-points estimators.

    Args:
        cfg (dict): Config of key-points detector.

    Returns:
        list: List of estimator.

    """
    # build an estimator
    def _build_estimator(layers_def):
        layers = []
        for l_def in layers_def:
            if isinstance(l_def, list):
                ksize, out_channels, pad, activ = l_def
                if activ == 'r':
                    layer = Conv2DActiv(
                        in_channels=None, out_channels=out_channels, ksize=ksize,
                        pad=pad, activ=relu)
                elif activ == 's':
                    layer = Conv2DActiv(
                        in_channels=None, out_channels=out_channels, ksize=ksize,
                        pad=pad, activ=sigmoid)
                elif activ == 'sr':
                    layer = Conv2DActiv(
                        in_channels=None, out_channels=out_channels, ksize=ksize,
                        pad=pad, activ=sigmoid_for_residual)
                elif activ == 'n':
                    layer = Convolution2D(
                        in_channels=None, out_channels=out_channels, ksize=ksize,
                        pad=pad)
                else:
                    raise ValueError('Unsupported type: {}'.format(activ))
            elif isinstance(l_def, str):
                if l_def == 'p':
                    layer = MaxPool2D()
                else:
                    raise ValueError('Unsupported type: {}'.format(type(l_def)))
            else:
                raise ValueError('Unsupported type: {}'.format(type(l_def)))

            layers.append(layer)
        estimator = chainer.Sequential(*layers)
        return estimator

    lst_estimator_def = cfg['estimator']
    lst_estimator = [_build_estimator(estimator_def)
                     for estimator_def in lst_estimator_def]
    return lst_estimator


def build_meta_architecture(cfg, extractor, lst_estimator):
    """ Build meta-architecture with feature extractor and estimators.

    Args:
        cfg (dict): Config of key-points detector.
        extractor (Chain): Feature extractor.
        lst_estimator (list): List of estimator.

    Returns:
        Chain: Model for key-points detector.

    """
    if cfg['meta_architecture'] == 'SingleStageNet':
        if len(lst_estimator) > 1:
            warn_msg = 'Expected number of estimator is 1. ' \
                  'Got {}.'.format(len(lst_estimator))
            warnings.warn(warn_msg)

        model = SingleStageNet(extractor, lst_estimator[0])
    elif cfg['meta_architecture'] == 'Stage3Net':
        if len(lst_estimator) < 2:
            warn_msg = 'Expected number of estimator is more than one.'
            warnings.warn(warn_msg)
        model = Stage3Net(extractor, lst_estimator)
    elif cfg['meta_architecture'] == 'ResidualStage3Net':
        if len(lst_estimator) < 2:
            warn_msg = 'Expected number of estimator is more than one.'
            warnings.warn(warn_msg)
        model = ResidualStage3Net(extractor, lst_estimator)
    elif cfg['meta_architecture'] == 'SubResidualStage3Net':
        if len(lst_estimator) < 2:
            warn_msg = 'Expected number of estimator is more than one.'
            warnings.warn(warn_msg)
        model = SubResidualStage3Net(extractor, lst_estimator)

    else:
        err_msg = 'Not support meta architecture: `{}`.'.format(
            cfg['meta_architecture'])
        raise NotImplementedError(err_msg)
    return model


def build_model(cfg):
    """ Build model for hand key-points detector.

    Args:
        cfg (dict): Config of key-points detector.

    Returns:
        Chain: Model of key-points detector.

    """
    extractor = build_extractor(cfg)
    lst_estimator = build_estimators(cfg)
    model = build_meta_architecture(cfg, extractor, lst_estimator)
    return model
