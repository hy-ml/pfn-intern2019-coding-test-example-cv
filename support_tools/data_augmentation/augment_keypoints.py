import os
from glob import glob
import csv
import argparse
import numpy as np

img_size = np.array((640, 640))
center = img_size / 2

lst_angle = [i * 30 for i in range(12)]
lst_angle_pi = [-i * 30/180 * np.pi for i in range(12)]
lst_scale = [0.5, 0.75, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', type=str,
                        help='Path to the directory of input key-points files.')
    parser.add_argument('outdir', type=str,
                        help='Path to the directory of output key-points files.')
    args = parser.parse_args()

    input_paths = sorted(glob(os.path.join(args.indir, '*')))

    for input_path in input_paths:
        data = np.loadtxt(input_path, delimiter=',')
        keypoints = data[:, :2]
        status = data[:, 2]
        keypoints_norm = keypoints - center

        basename, ext = os.path.splitext(os.path.basename(input_path))
        output_path_tmplt = os.path.join(
            args.outdir, basename + '_{}_{}' + ext)
        for angle, angle_pi in zip(lst_angle, lst_angle_pi):
            rot = np.array(
                [[np.cos(angle_pi), -np.sin(angle_pi)],
                 [np.sin(angle_pi), np.cos(angle_pi)]])
                # [[-np.sin(angle_pi), np.cos(angle_pi)],
                #  [np.cos(angle_pi), np.sin(angle_pi)]])
            for scale in lst_scale:
                keypoints_aug = keypoints_norm * scale
                keypoints_aug = np.transpose(keypoints_aug, (1, 0))
                keypoints_aug = \
                    np.matmul(rot, keypoints_aug).transpose((1, 0)) + center
                # keypoints_aug = \
                #     np.matmul(rot, keypoints_norm.transpose(1, 0))
                # keypoints_aug = keypoints_aug * scale
                # keypoints_aug = keypoints_aug.transpose((1, 0)) + center

                output_path = output_path_tmplt.format(round(angle, 2), round(scale, 2))

                with open(output_path, 'w') as fp:
                    writer = csv.writer(fp)
                    for pt, st in zip(keypoints_aug, status):
                        pt = [int(p) for p in pt]
                        writer.writerow(list(pt) + [int(st)])


if __name__ == '__main__':
    main()

