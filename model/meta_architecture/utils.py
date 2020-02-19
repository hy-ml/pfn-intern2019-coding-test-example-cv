def concat_features_vgg(x):
    raise NotImplementedError()


def concat_features_resnet(x):
    raise NotImplementedError()


def concat_features_googlenet(x):
    raise NotImplementedError()


def concat_features(x, extractor_name):
    if 'vgg' in extractor_name:
        for k, v in x.items():
            return v
    elif 'resnet' in extractor_name:
        h = concat_features_resnet(x)
    elif 'googlenet' in extractor_name:
        h = concat_features_googlenet(x)
    else:
        raise NotImplementedError('Not support {}.'.format(extractor_name))
    return h
