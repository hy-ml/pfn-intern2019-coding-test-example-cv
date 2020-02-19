import os
import argparse
from glob import glob
import yaml
from annotator import Annotator
from annotator.utils import make_edge_table


fingers = ['thumb', 'pointer_finger', 'middle_finger', 'ring_finger',
           'little_finger']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='./config/annotation/16pts.yml',
                        help='Path to the config YAML file.')
    args = parser.parse_args()
    return args


def cvt_ext_to_csv(file_path):
    """

    Args:
        file_path (str): Path to the file like aaa/bbb/ccc.txt.

    Returns:
        str: Path converted to csv.

    """
    file_path_without_extension = file_path[:file_path.rindex('.')] + '.csv'
    return file_path_without_extension


def main():
    args = parse_args()
    with open(args.config, 'r') as fp:
        # cfg = yaml.load(fp, Loader=yaml.FullLoader)
        cfg = yaml.load(fp)

    palm_indices = cfg['palm']
    lst_finger_indices = [cfg[finger] for finger in fingers]
    edge_table = make_edge_table(palm_indices, lst_finger_indices)

    if 'guide_img' in cfg.keys() and 'guide_keypoints' in cfg.keys():
        annotator = Annotator(cfg['n_keypoints'], edge_table,
                              cfg['guide_img'], cfg['guide_keypoints'])
    else:
        annotator = Annotator(cfg['n_keypoints'], edge_table, None, None)

    img_paths = sorted(glob(os.path.join(cfg['img_dir'], '*')))
    save_paths = \
        [os.path.join(cfg['save_dir'], cvt_ext_to_csv(os.path.basename(p)))
         for p in img_paths]

    for i, (img_path, save_path) in enumerate(zip(img_paths, save_paths)):
        print('{} / {}'.format(i+1, len(img_paths)))
        if os.path.isfile(save_path):
            continue
        is_continue = annotator(img_path, save_path)
        if not is_continue:
            break


if __name__ == '__main__':
    main()
