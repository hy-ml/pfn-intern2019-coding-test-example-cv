import os
import argparse
from copy import deepcopy
from glob import glob
import numpy as np
import cv2


lst_angle = [i * 30 for i in range(12)]
lst_scale = [0.5, 0.75, 1]


def separate_background(img):
    img = deepcopy(img)
    img_f = cv2.medianBlur(img, ksize=7)
    img_f = img
    img_b = img[:, :, 0]
    img_g = img[:, :, 1]
    img_r = img[:, :, 2]
    img_g_f = img_f[:, :, 1]
    img_r_f = img_f[:, :, 2]

    img_b_s = np.where((img_r_f > 80) & (img_g_f > 20), img_b, 0)
    img_g_s = np.where((img_r_f > 80) & (img_g_f > 20), img_g, 0)
    img_r_s = np.where((img_r_f > 80) & (img_g_f > 20), img_r, 0)

    img[:,:,0] = img_b_s
    img[:,:,1] = img_g_s
    img[:,:,2] = img_r_s
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', type=str,
                        help='Path to the directory of input images.')
    parser.add_argument('outdir', type=str,
                        help='Path to the directory of output images.')
    args = parser.parse_args()

    input_paths = sorted(glob(os.path.join(args.indir, '*')))

    for i, input_path in enumerate(input_paths):
        img = cv2.imread(input_path)
        h, w, _ = img.shape
        center = (h/2, w/2)

        # output_path = os.path.join(args.outdir, os.path.basename(input_path))
        # cv2.imwrite(output_path, img)
        img_separated = separate_background(img)
        img_basename, ext = os.path.splitext(os.path.basename(input_path))
        output_path_tmplt = os.path.join(
            args.outdir, img_basename + '_{}_{}' + ext)
        for angle in lst_angle:
            for scale in lst_scale:
                trans = cv2.getRotationMatrix2D(center, angle, scale)
                img_aug = cv2.warpAffine(img_separated, trans, (w, h))
                output_path = output_path_tmplt.format(angle, round(scale, 2))
                cv2.imwrite(output_path, img_aug)


if __name__ == '__main__':
    main()
