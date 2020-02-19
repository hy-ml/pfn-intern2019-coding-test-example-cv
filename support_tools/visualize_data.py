"""
Visualize data.
"""
import os
from glob import glob
import argparse
import numpy as np
import cv2


def draw_points(img, points, color=(255, 0, 0)):
    """

    Args:
        img (np.ndarray): Array of an image.
        points (list or PointList): An instance contains key-points.
        color (tuple): BGR color.

    """

    for p in points:
        x, y = p
        cv2.circle(img, (x, y), 2, color, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir', type=str,
                        help='Path to the directory of inpout images.')
    parser.add_argument('keypoints_dir', type=str,
                        help='Path to the directory of input keypoints files.')
    args = parser.parse_args()

    img_paths = sorted(glob(os.path.join(args.img_dir, '*')))
    keypoints_paths = sorted(glob(os.path.join(args.keypoints_dir, '*')))
    for img_path, keypoints_path in zip(img_paths, keypoints_paths):
        img = cv2.imread(img_path)
        data = np.loadtxt(keypoints_path, delimiter=',').astype(np.int)
        keypoints = (data[:, :2]).tolist()
        print('============================')
        print(os.path.basename(img_path))
        print(os.path.basename(keypoints_path))
        draw_points(img, keypoints)
        cv2.imshow('img', img)
        key = cv2.waitKey(0) & 0xff
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
