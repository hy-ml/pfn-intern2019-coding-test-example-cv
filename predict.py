import os
import argparse
import yaml
from glob import glob
import numpy as np
from PIL import Image
import cv2
import cupy
import chainer
from chainer.backend import  cuda
from model import build_model, KeypointsDetector
from data.transforms import build_transforms
from annotator.utils import draw_points, draw_edge
from annotator.utils import make_edge_table


fingers = ['thumb', 'pointer_finger', 'middle_finger', 'ring_finger',
           'little_finger']


def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_detector', type=str,
                        help='Path to the config file of pose estimation.')
    parser.add_argument('config_annotation')
    parser.add_argument('snapshot', type=str,
                        help='Path to the checkpoint file.')
    parser.add_argument('img_dir')
    parser.add_argument('device', type=int)
    parser.add_argument('--visualize_mode', type=str, default='keypoints',
                        choices=['keypoints', 'heatmap'])
    args = parser.parse_args()

    with open(args.config_detector, 'r') as fp:
        cfg_detector = yaml.load(fp)

    with open(args.config_annotation, 'r') as fp:
        cfg_anno = yaml.load(fp)
    palm_indices = cfg_anno['palm']
    lst_finger_indices = [cfg_anno[finger] for finger in fingers]
    edge_table = make_edge_table(palm_indices, lst_finger_indices)

    detector = build_model(cfg_detector)
    chainer.serializers.load_npz(args.snapshot, detector,
                                 path='updater/model:main/predictor/')
    if args.device >= 0:
        chainer.cuda.get_device_from_id(args.device).use()  # Make a specified GPU current
        detector.to_gpu(args.device)
    transforms = build_transforms((224, 224), None, mode='test')

    img_paths = sorted(glob(os.path.join(args.img_dir, '*')))
    chainer.using_config('trian', False)
    for img_path in img_paths:
        img = Image.open(img_path)
        x, _ = transforms(img, None)
        x = np.expand_dims(x, axis=0)
        if args.device >= 0:
            x = cuda.to_gpu(x, device=args.device
                              )
        heatmaps = detector(x)
        if isinstance(heatmaps, list):
            heatmaps = heatmaps[-1]
        heatmaps = heatmaps.data
        if args.device >= 0:
            heatmaps = cupy.asnumpy(heatmaps)
        heatmaps_filtered = exec_gaussian_filter(heatmaps)
        img = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)

        visualize_keypoints(heatmaps_filtered.astype(np.uint8), img, edge_table)


def normalize_array(x):
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x)) * 255
    return x_norm


def exec_gaussian_filter(heatmaps, sigma=0):
    heatmaps = np.transpose(np.squeeze(heatmaps), (1, 2, 0))
    heatmaps_norm = normalize_array(heatmaps).astype(np.uint8)
    heatmaps_filtered = cv2.GaussianBlur(heatmaps_norm, (3, 3), sigma)
    heatmaps_filtered = np.transpose(heatmaps_filtered, (2, 0, 1))
    return heatmaps_filtered


def visualize_keypoints(heatmaps, img, edge_table):
    keypoints = []
    for heatmap in heatmaps:
        pt = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        y, x = [int(v * 640 / 56) for v in pt]
        keypoints.append((x, y))
    draw_points(img, keypoints)
    draw_edge(img, keypoints, edge_table)
    cv2.imshow('Predict', img)
    key = cv2.waitKey(0) & 0xff
    if key == ord('q'):
        exit()


if __name__ == '__main__':
    predict()
