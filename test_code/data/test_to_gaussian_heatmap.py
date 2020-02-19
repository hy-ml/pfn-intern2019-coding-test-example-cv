import sys
import argparse
import numpy as np
from PIL import Image
import cv2
sys.path.append('../..')
from data.transforms import ToGaussianHeatmap, ResizeKeypoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path')
    parser.add_argument('keypoints_path')
    args = parser.parse_args()

    img = Image.open(args.img_path)
    keypoints = np.loadtxt(args.keypoints_path, delimiter=',')[:,:2]

    # to_gaussian_map = ToGaussianHeatmap((640, 640), 150)
    to_gaussian_map = ToGaussianHeatmap((224, 224), 100)
    resize_keypoints = ResizeKeypoints((224, 224))
    img, keypoints = resize_keypoints(img,  keypoints)
    _, heatmaps = to_gaussian_map(None, keypoints)
    img = cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))

    cv2.imshow('RGB', img)

    for heatmap in heatmaps:
        heatmap = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (224, 224))
        cv2.imshow('Heatmap', heatmap)
        cv2.waitKey(0)


if __name__== '__main__':
    main()
