from copy import deepcopy
import csv
import numpy as np
import cv2
from annotator.utils import PointList, draw_points, draw_edge


class Annotator(object):
    def __init__(self, n_points, edge_table, guide_img_path=None,
                 guide_keypoints_path=None):
        self.n_points = n_points
        self.edge_table = edge_table

        self.annot_wname = "Annotation Window"
        cv2.namedWindow(self.annot_wname)

        if guide_img_path is not None and guide_keypoints_path is not None:
            self.guide_wname = "Annotation Guide"
            cv2.namedWindow(self.guide_wname)

            self.guide_keypoints = PointList(16)
            self.guide_keypoints.ptlist = (np.loadtxt(
                guide_keypoints_path, dtype=np.int, delimiter=',')[:,:2]).tolist()
            self.guide_img = cv2.imread(guide_img_path)
        else:
            self.guide_img = None

    def __call__(self, img_path, save_path):
        """

        Args:
            img_path (str): Path to the directory of input images.
            save_path (str): Path to the directory of csv files saved.


        """
        img = cv2.imread(img_path)
        img_org = deepcopy(img)
        keypoints = PointList(self.n_points)
        cv2.setMouseCallback(self.annot_wname, self._on_mouse,
                             [self.annot_wname, img, keypoints])
        self._update_guide(0)

        while True:
            cv2.imshow(self.annot_wname, img)
            key = cv2.waitKey(10) & 0xFF

            if key == ord('q'):
                cv2.destroyAllWindows()
                return False
            elif key == ord('n'):
                return True
            elif key == ord('b'):
                keypoints.sub()
                img = deepcopy(img_org)
                draw_points(img, keypoints)
                draw_edge(img, keypoints, self.edge_table)
                cv2.setMouseCallback(self.annot_wname, self._on_mouse,
                                     [self.annot_wname, img, keypoints])
                self._update_guide(len(keypoints.ptlist))
            elif key == ord('s'):
                if self._save_key_points_as_csv(keypoints, save_path):
                    print('Saved key-points to {}.'.format(save_path))
                self._update_guide(keypoints.len_ptlist())
            elif key == ord('o'):
                keypoints.add(-1, -1, -1)
                self._update_guide(keypoints.len_ptlist())

    def _save_key_points_as_csv(self, keypoints, save_path):
        if len(keypoints.ptlist) != self.n_points:
            print('Not enough points specified')
            return False
        with open(save_path, 'w') as fp:
            writer = csv.writer(fp)
            for pt, st in zip(keypoints.ptlist, keypoints.stlist):
                writer.writerow(list(pt) + [st])
        return True

    def _on_mouse(self, event, x, y, _, params):
        _, img, keypoints = params

        if event == cv2.EVENT_LBUTTONDOWN:
            if keypoints.add(x, y, 0):
                draw_points(img, keypoints)
                draw_edge(img, keypoints, self.edge_table)
                self._update_guide(keypoints.len_ptlist())
        elif event == cv2.EVENT_RBUTTONDOWN:
            if keypoints.add(x, y, 1):
                draw_points(img, keypoints)
                draw_edge(img, keypoints, self.edge_table)
                self._update_guide(keypoints.len_ptlist())

    def _update_guide(self, annot_idx):
        # when annotation guide is not necessary
        if self.guide_img is None:
            return

        draw_points(self.guide_img, self.guide_keypoints)
        draw_edge(self.guide_img, self.guide_keypoints, self.edge_table)
        # when all key points is not annotated
        if annot_idx < self.n_points:
            cv2.circle(self.guide_img, tuple(self.guide_keypoints.ptlist[annot_idx]),
                       2, (0, 0, 255), 2)
        cv2.imshow(self.guide_wname, self.guide_img)
