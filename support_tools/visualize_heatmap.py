"""
Visualize estimated heatmap.
"""
import os
import sys
import argparse
from copy import deepcopy
import yaml
import numpy as np
from PIL import Image
import cv2
import chainer
from chainer.backend import  cuda
sys.path.append('..')
from model import build_model
from data.transforms import build_transforms
from data.keypoints_dataset import KeypointsDataset
from annotator.utils import draw_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_detector', type=str,
                        help='Path to the config file of pose estimation.')
    parser.add_argument('snapshot', type=str,
                        help='Path to the checkpoint file.')
    parser.add_argument('img_dir', type=str,
                        help='Path to the directory of input images.')
    parser.add_argument('gpu', type=int,
                        help='GPU ID.')
    args = parser.parse_args()

    with open(args.config_detector, 'r') as fp:
        cfg_detector = yaml.load(fp)

    detector = build_model(cfg_detector)
    chainer.serializers.load_npz(args.snapshot, detector, path='updater/model:main/predictor/')
    detector.to_gpu(args.gpu)
    transforms = build_transforms((224, 224), None)
    dataset = KeypointsDataset(
        os.path.expanduser(os.path.join('..', cfg_detector['train_img_dir'] + '_a')),
        os.path.expanduser(os.path.join('..', cfg_detector['train_keypoints_dir'] + '_a')), transforms)

    for img_path, keypoints_path in zip(
            dataset._img_paths, dataset._keypoints_paths):
        img = Image.open(img_path)
        x, _ = transforms(img, None)
        x = np.expand_dims(x, axis=0)
        keypoints = np.loadtxt(keypoints_path, dtype=np.int, delimiter=',')
        if args.gpu >= 0:
            x = cuda.to_gpu(x, device=args.gpu
                              )
        heatmaps = np.squeeze(detector(x).array)
        img = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        if args.gpu >= 0:
            heatmaps = heatmaps.get()

        visualize_heatmap(img, heatmaps, keypoints)


def visualize_heatmap(img, heatmaps, keypoints):
    for heatmap, keypoint in zip(heatmaps, keypoints):
        img_with_keypoint = deepcopy(img)
        x, y = keypoint[:2]
        draw_points(img_with_keypoint, [(x, y)])

        # Normalize heatmap
        # heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap)) * 255
        heatmap = np.where(heatmap < 0, 0, heatmap)
        heatmap = 255 * (heatmap / np.max(heatmap))

        heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (640, 640))
        cv2.imshow('Heatmap', heatmap)
        cv2.imshow('RGB', img_with_keypoint)
        key = cv2.waitKey(0) & 0xff
        if key == ord('n'):
            return
        elif key == ord('q'):
            exit()
        else:
            continue


if __name__ == '__main__':
   main()

