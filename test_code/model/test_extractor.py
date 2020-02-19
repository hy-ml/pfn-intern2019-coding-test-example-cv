"""
This program test whether transform function transforms data properly and
build extractor pre-trained by ImageNet properly.
This program require images whose file name is
`category name in ImageNet + file extension`. For example,
file name should be tiger.jpg if its category is tiger.
"""

import os
import sys
import argparse
from glob import glob
import yaml
import numpy as np
from PIL import Image
from chainer import cuda
sys.path.append('../..')
from test_code.imagenet_categories import imagenet_categories
from model.build import build_extractor
from data.transforms import ResizeImg, ToNumpyWithMeanNorm, Compose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        help='Path to the config file of pose estimation.')
    parser.add_argument('--img_dir', type=str, default='../imgs/imagenet',
                        help='Path to the directory of input images. Default '
                             'is `../imgs/imagenet`.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Using GPU ID. Default is 0. -1 means only using CPU.')
    args = parser.parse_args()

    with open(args.config, 'r') as fp:
        cfg = yaml.load(fp)

    classifier = build_extractor(cfg)
    if args.gpu >= 0:
        classifier.to_gpu(device=args.gpu)
    transforms = Compose([
        ResizeImg((224, 224)),
        ToNumpyWithMeanNorm(cfg['extractor'], (224, 224)),
    ])

    img_paths = glob(os.path.join(args.img_dir, '*'))

    for img_path in img_paths:
        img = Image.open(img_path)
        x, _ = transforms(img, None)
        x = np.expand_dims(x, axis=0)
        if args.gpu >= 0:
            x = cuda.to_gpu(x, device=args.gpu)

        y = np.squeeze(classifier(x)['prob'].data)
        if args.gpu >= 0:
            y = cuda.to_cpu(y)

        categories_predict = imagenet_categories[np.argmax(y)]

        category_annot, _ = os.path.splitext(os.path.basename(img_path))
        print('Annotation: {}, Predict: {}'.format(category_annot, categories_predict))
        assert category_annot in categories_predict

    print('Success Test')


if __name__ == '__main__':
    main()
