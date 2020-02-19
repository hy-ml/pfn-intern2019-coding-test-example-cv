import os
import sys
import argparse
from copy import deepcopy
import numpy as np
import cv2
import yaml
sys.path.append('../..')
from data.transforms import build_transforms
from data.keypoints_dataset import KeypointsDataset


def test_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()

    with open(args.config, 'r') as fp:
        # cfg = yaml.load(fp, Loader=yaml.FullLoader)
        cfg = yaml.load(fp)

    img_dir = os.path.expanduser(cfg['train_img_dir'])
    keypoints_dir = os.path.expanduser(cfg['train_keypoints_dir'])

    transforms = build_transforms((224, 224), (54, 54), 'vgg16')
    dataset = KeypointsDataset(img_dir, keypoints_dir, transforms)

    for data in dataset:
        img, heatmaps = data
        img_b, img_g, img_r = [np.expand_dims(img_ - np.min(img_), axis=-1) for img_ in img]
        img = np.concatenate((img_b, img_g, img_r), axis=-1).astype(np.uint8)

        for heatmap in heatmaps:
            heatmap = heatmap * 255
            heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
            # heatmap = cv2.resize(heatmap, (224, 224))
            cv2.imshow('Heatmap', heatmap)
            cv2.imshow('RGB', img)
            key = cv2.waitKey(0) & 0xff
            if key == ord('n'):
                break
            elif key == ord('q'):
                exit()
            else:
                continue


if __name__ == '__main__':
    test_dataset()
