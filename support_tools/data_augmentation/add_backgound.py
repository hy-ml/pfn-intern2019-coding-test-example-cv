import os
from glob import glob
import argparse
import random
from copy import deepcopy
import numpy as np
import cv2


def crop_img(img):
    h, w, _ = img.shape
    if h == w:
        crop_img = deepcopy(img)
    elif h > w:
        start_h = random.randint(0, h-w)
        crop_img = img[start_h:start_h+w, :, :]
    else:
        start_w = random.randint(0, w-h)
        crop_img = img[:, start_w:start_w+h, :]
    return crop_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('hand_img_dir', type=str,
                        help='Path to the directory of input hand images.')
    parser.add_argument('background_img_dir', type=str,
                        help="Path to the directory of input background images.")
    parser.add_argument('outdir', type=str,
                        help='Path to the output directory.')
    args = parser.parse_args()

    hand_img_paths = glob(os.path.join(args.hand_img_dir, '*'))
    background_img_paths = glob(os.path.join(args.background_img_dir, '*'))
    background_imgs = [crop_img(cv2.imread(p)) for p in background_img_paths]

    for hand_img_path in hand_img_paths:
        hand_img = cv2.imread(hand_img_path)
        background_img = \
            background_imgs[random.randint(0, len(background_imgs)-1)]

        # merged_img = \
        #     np.where(np.all(hand_img==0, axis=-1), hand_img, background_img)
        hand_img_chs = [img_ch for img_ch in np.transpose(hand_img, (2, 0, 1))]
        background_img_chs = \
            [img_ch for img_ch in np.transpose(background_img, (2, 0, 1))]
        merged_img_chs = []
        for ch in range(3):
            merged_img_ch = \
                np.expand_dims(np.where(np.all(hand_img == 0, axis=-1),
                         background_img_chs[ch], hand_img_chs[ch]), axis=2)
            merged_img_chs.append(merged_img_ch)
        merged_img = np.concatenate(merged_img_chs, axis=2)
        merged_img = cv2.medianBlur(merged_img, 3)
        output_path = os.path.join(args.outdir, os.path.basename(hand_img_path))
        cv2.imwrite(output_path, merged_img)


if __name__ == '__main__':
    main()
