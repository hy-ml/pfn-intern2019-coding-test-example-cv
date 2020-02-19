import os
from glob import glob
import argparse
import cv2


resize_size = 640


def resize(img):
    h, w, _ = img.shape

    if h >= w:
        resize_h = int(h * resize_size / w)
        resize_w = resize_size
    else:
        resize_h = resize_size
        resize_w = int(w * resize_size / h)

    img_reize = cv2.resize(img, (resize_w, resize_h))
    return img_reize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', type=str,
                        help='Path to the directory of input images')
    parser.add_argument('outdir', type=str,
                        help='Path to the directory of output images.')
    args = parser.parse_args()

    input_paths = glob(os.path.join(args.indir, '*'))

    for input_path in input_paths:
        output_path = os.path.join(args.outdir, os.path.basename(input_path))
        img = cv2.imread(input_path)
        resize_img = resize(img)
        cv2.imwrite(output_path, resize_img)


if __name__ == '__main__':
    main()
