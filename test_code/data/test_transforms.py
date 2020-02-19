import sys
import argparse
from copy import deepcopy
import numpy as np
import yaml
from PIL import Image
import cv2
sys.path.append('../..')
from annotator.utils import draw_points, draw_edge, make_edge_table
from data.transforms import ResizeImg, ResizeKeypoints, ToHeatmap


fingers = ['thumb', 'pointer_finger', 'middle_finger', 'ring_finger',
           'little_finger']


def test_transforms():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('img')
    parser.add_argument('keypoints')
    args = parser.parse_args()

    with open(args.config, 'r') as fp:
        # cfg = yaml.load(fp, Loader=yaml.FullLoader)
        cfg = yaml.load(fp)
    palm_indices = cfg['palm']
    lst_finger_indices = [cfg[finger] for finger in fingers]

    edge_table = make_edge_table(palm_indices, lst_finger_indices)
    img = Image.open(args.img)
    keypoints_data = np.loadtxt(args.keypoints, delimiter=',', dtype=np.int)
    keypoints = keypoints_data[:, :2]

    # --------------------------------------------------------------------------
    # Test ResizeImg, ResizeKeypoints
    # --------------------------------------------------------------------------
    print('Test ResizeImg and ResizeKeypoints')
    resize_img = ResizeImg((224, 224))
    resize_keypoints = ResizeKeypoints((224, 224))

    img_org = deepcopy(img)
    keypoints_org = deepcopy(keypoints)

    img, keypoints = resize_keypoints(img, keypoints)
    img, keypoints = resize_img(img, keypoints)

    img = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    keypoints = keypoints.tolist()
    img_org = cv2.cvtColor(np.asarray(img_org, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    keypoints_org = keypoints_org.tolist()

    draw_points(img, keypoints)
    draw_edge(img, keypoints, edge_table)
    draw_points(img_org, keypoints_org)
    draw_edge(img_org, keypoints_org, edge_table)

    cv2.imshow('Test ResizeImg and ResizeKeypoints: Original', img_org)
    cv2.imshow('Test ResizeImg and ResizeKeypoints: Resize', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --------------------------------------------------------------------------
    # Test ToHeatmap
    # --------------------------------------------------------------------------
    print('Test ToHeatmap')
    img_transposed = np.transpose(img, (2, 0, 1))
    to_heatmap = ToHeatmap()
    _, heatmap = to_heatmap(img_transposed, keypoints)
    heatmap = np.sum(heatmap * 255, axis=0).astype(np.uint8)
    cv2.imshow('Test ToHeatmap: RGB Image', img)
    cv2.imshow('Test ToHeatmap: Heatmap', heatmap)
    cv2.waitKey(0)


if __name__ == '__main__':
    test_transforms()
