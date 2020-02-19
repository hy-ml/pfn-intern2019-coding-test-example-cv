import random
import numpy as np
from PIL import ImageOps


def build_transforms(resize_size_img, resize_size_keypoints, model='vgg',
                     mode='train'):
    if mode == 'train' or mode == 'val':
        transforms = Compose([
            # FlipImgKeypoints(),
            ResizeKeypoints(resize_size_keypoints),
            ResizeImg(resize_size_img),
            ToNumpyWithMeanNorm(model, resize_size_img),
            ToGaussianHeatmap(resize_size_keypoints, 5),
        ])
    elif mode == 'test':
        transforms = Compose([
            ResizeImg(resize_size_img),
            ToNumpyWithMeanNorm(model, resize_size_img),
        ])
    else:
        raise ValueError('Not support mode {}.'.format(mode))
    return transforms


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, keypoints):
        for t in self.transforms:
            img, keypoints = t(img, keypoints)
        return img, keypoints


class FlipImgKeypoints(object):
    def __call__(self, img, keypoints):
        if random.random() > 0.5:
            return img, keypoints

        img_flip = ImageOps.mirror(img)

        w, _ = img.size
        keypoints_flip = []
        for pt in keypoints:
            x, y = pt
            x_flip = w - x - 1
            keypoints_flip.append([x_flip, y])

        return img_flip, keypoints_flip


class ResizeImg(object):
    def __init__(self, resize_size):
        self.resize_size = resize_size

    def __call__(self, img, keypoints):
        return img.resize(self.resize_size), keypoints


class ResizeKeypoints(object):
    def __init__(self, resize_size):
        self.resize_size = resize_size

    def __call__(self, img, keypoints):
        if keypoints is None:
            return img, None
        w, h = img.size
        resize_w, resize_h = self.resize_size
        scale_w, scale_h = resize_w / w, resize_h / h

        resize_keypoints = keypoints * np.array([scale_w, scale_h])
        resize_keypoints = resize_keypoints.astype(np.int)
        return img, resize_keypoints


class ToNumpyWithMeanNorm(object):
    def __init__(self, model, resize_size):
        if 'vgg' in model:
            from chainer.links.model.vision.vgg import prepare
        elif 'resnet' in model:
            from chainer.links.model.vision.resnet import prepare
        elif 'googlenet' in model:
            from chainer.links.model.vision.googlenet import prepare
        else:
            raise NotImplementedError('Not support `{}`.'.format(model))
        self._prepare = prepare
        self._resize_size = resize_size

    def __call__(self, img, keypoints):
        img = self._prepare(img, self._resize_size)
        return img, keypoints


class ToHeatmap(object):
    def __init__(self, heatmap_size):
        if heatmap_size is not None:
            self.width, self.height = heatmap_size

    def __call__(self, img, keypoints):
        if keypoints is None:
            return img, None

        heatmap = np.zeros((len(keypoints), self.height, self.width), dtype=np.float32)

        for i, pt in enumerate(keypoints):
            x, y = pt
            heatmap[i, y, x] = 1
        return img, heatmap


class ToGaussianHeatmap(object):
    def __init__(self, heatmap_size, sigma):
        if heatmap_size is not None:
            self.width, self.height = heatmap_size
        self.sigma = sigma

    def __call__(self, img, keypoints):
        if keypoints is None:
            return img, None

        cord_map = np.array(
            [(y, x) for y in range(self.height) for x in range(
                self.width)]).reshape((self.height, self.width, 2))

        heatmaps = []
        for keypoint in keypoints:
            x, y = keypoint[:2]
            distance_map = cord_map - np.array((y, x))
            heatmap = self._calc_gaussian(np.sum(np.square(distance_map), axis=-1))
            heatmaps.append(np.expand_dims(heatmap, axis=0))
        heatmaps = np.concatenate(tuple(heatmaps), axis=0).astype(np.float32)

        return img, heatmaps

    def _calc_gaussian(self, x):
        return np.exp(-1 * np.square(x) / np.square(self.sigma))


