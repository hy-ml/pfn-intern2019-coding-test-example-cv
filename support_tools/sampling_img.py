"""
Sampling images.
"""
import os
import argparse
import shutil
from glob import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_img_dir', type=str,
                        help='Path to the directory of input images.')
    parser.add_argument('output_img_dir', type=str,
                        help='Path to the directory of output images.')
    parser.add_argument('sampling_step', type=int,
                        help='The number of sampling step.')
    args = parser.parse_args()

    img_paths = sorted(glob(os.path.join(args.input_img_dir, '*')))
    sampled_img_paths = img_paths[::args.sampling_step]

    for img_path in sampled_img_paths:
        save_path = os.path.join(args.output_img_dir, os.path.basename(img_path))
        shutil.copy(img_path, save_path)


if __name__ == '__main__':
    main()
