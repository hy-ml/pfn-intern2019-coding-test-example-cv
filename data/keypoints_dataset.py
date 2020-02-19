import os
from glob import glob
from PIL import Image
import numpy as np
from chainer import dataset


class KeypointsDataset(dataset.DatasetMixin):
    def __init__(self, img_dir, keypoints_dir, transforms, img_ext='.png'):
        """

        Args:
            img_dir (str): Path to the directory of input images.
            keypoints_dir: Path to the directory of keypoints csv files.
            transforms (Compose): Transform class. Check `data/transforms.py`
            img_ext (str): Extension of input images.
        """
        super(KeypointsDataset, self).__init__()
        self._keypoints_paths = sorted(glob(os.path.join(keypoints_dir, '*')))

        self._img_paths = []
        for keypoints_path in self._keypoints_paths:
            basename, _ = os.path.splitext(os.path.basename(keypoints_path))
            img_path = os.path.join(img_dir, basename + img_ext)
            if not os.path.isfile(img_path):
                raise ValueError('Not exists image: {}.'.format(img_path))
            self._img_paths.append(img_path)
        self._transforms = transforms

    def get_example(self, i):
        img = Image.open(self._img_paths[i])
        keypoints_data = np.loadtxt(self._keypoints_paths[i], delimiter=',',
                               dtype=np.int)
        keypoints = keypoints_data[:, :2]
        status = keypoints_data[:, 2]
        img, heatmap = self._transforms(img, keypoints)
        return img, heatmap

    def __len__(self):
        return len(self._img_paths)
